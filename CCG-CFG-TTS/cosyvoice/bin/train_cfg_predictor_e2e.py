#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function

import sys, os
sys.path.append('third_party/Matcha-TTS')
sys.path.append('./')

import argparse
import time
import logging
import json
import glob
import ipdb
from copy import deepcopy
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np



from cosyvoice.models.cfg_predictor import CFGScalePredictor, apply_guided_sampling
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.train_utils import init_distributed, wrap_cuda_model


# # For debug, setting dist variables
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '0'
# os.environ['LOCAL_RANK'] = '0'



class EndToEndCFGDataset(Dataset):
    """
    Dataset for end-to-end CFG predictor training.
    Instead of pre-extracted logits, we store the input data and compute
    conditional/unconditional logits during training.
    """
    
    def __init__(self, data_file: str, max_seq_len: int = 368):
        self.max_seq_len = max_seq_len
        
        # Load ground truth data from PyTorch file
        # Expected format: a list of dictionaries with:
        # {
        #   "text": "Hello world",
        #   "prompt_text": "Please generate speech",
        #   "ground_truth_tokens": tensor([123, 456, 789, ...]),
        #   "audio_path": "/path/to/audio.wav",
        #   "audio_id": "sample_001"
        # }
        
        self.samples = torch.load(data_file)
        
        # Filter samples by length
        self.samples = [
            sample for sample in self.samples 
            if len(sample['ground_truth_tokens']) <= max_seq_len
        ]
        
        logging.info(f"Loaded {len(self.samples)} samples from {data_file}")
        
        # Calculate word counts for each sample
        self.calculate_word_counts()
        
        # Analyze token length distribution
        self.analyze_token_length_distribution()
    
    def calculate_word_counts(self):
        """Calculate and store the word counts for text and prompt text."""
        for sample in self.samples:
            # Count words in target text (split by whitespace)
            target_text = sample.get('text', '')
            target_word_count = len(target_text.split()) if target_text else 0
            
            # Count words in prompt text
            prompt_text = sample.get('prompt_text', '')
            prompt_word_count = len(prompt_text.split()) if prompt_text else 0
            
            # Store the total word count
            sample['word_count'] = target_word_count + prompt_word_count
        
        # Log word count statistics
        word_counts = [sample['word_count'] for sample in self.samples]
        if word_counts:
            min_words = min(word_counts)
            max_words = max(word_counts)
            avg_words = sum(word_counts) / len(word_counts)
            median_words = sorted(word_counts)[len(word_counts) // 2]
            
            logging.info(f"Word Count Statistics:")
            logging.info(f"  Min: {min_words}, Max: {max_words}, Avg: {avg_words:.2f}, Median: {median_words}")
    
    def analyze_token_length_distribution(self):
        """Analyze and log the distribution of ground truth token lengths."""
        if not self.samples:
            logging.info("No samples to analyze token length distribution.")
            return
        
        # Get all lengths
        lengths = [len(sample['ground_truth_tokens']) for sample in self.samples]
        
        # Calculate statistics
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / len(lengths)
        median_len = sorted(lengths)[len(lengths) // 2]
        
        # Create a histogram of lengths
        hist_bins = 10
        bin_width = (max_len - min_len) / hist_bins if max_len > min_len else 1
        bins = {}
        
        for length in lengths:
            bin_idx = min(hist_bins - 1, int((length - min_len) / bin_width)) if bin_width > 0 else 0
            bin_start = min_len + bin_idx * bin_width
            bin_end = min_len + (bin_idx + 1) * bin_width
            bin_key = f"{int(bin_start)}-{int(bin_end)}"
            
            if bin_key in bins:
                bins[bin_key] += 1
            else:
                bins[bin_key] = 1
        
        # Log the results
        logging.info(f"Ground Truth Token Length Distribution:")
        logging.info(f"  Min: {min_len}, Max: {max_len}, Avg: {avg_len:.2f}, Median: {median_len}")
        logging.info(f"  Distribution by length ranges:")
        
        for bin_key, count in sorted(bins.items(), key=lambda x: int(x[0].split('-')[0])):
            percentage = (count / len(lengths)) * 100
            bar = "#" * int(percentage / 2)
            logging.info(f"  {bin_key}: {count} samples ({percentage:.1f}%) {bar}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'text': sample.get('text', ''),
            'prompt_text': sample.get('prompt_text', ''),
            'ground_truth_tokens': sample['ground_truth_tokens'],
            'audio_path': sample.get('audio_path', ''),
            'audio_id': sample.get('audio_id', f'sample_{idx}'),
            'seq_len': len(sample['ground_truth_tokens']),
            'word_count': sample.get('word_count', 0)  # Include word count
        }


class TokenBucketBatchSampler:
    """
    A batch sampler that creates batches based on token count rather than sample count.
    This ensures each batch has approximately the same computational load.
    
    This sampler works with DistributedDataParallel for multi-GPU training.
    """
    
    def __init__(self, dataset, batch_token_size=3000, batch_word_size=None, max_batch_size=32, min_batch_size=1, 
                 num_replicas=1, rank=0, shuffle=True, drop_last=False, seed=42):
        """
        Initialize a token-based batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_token_size: Target number of tokens per batch
            batch_word_size: Target number of words per batch (if specified, overrides batch_token_size)
            max_batch_size: Maximum number of samples in a batch regardless of token count
            min_batch_size: Minimum number of samples in a batch
            num_replicas: Number of distributed training workers
            rank: Rank of this worker in distributed training
            shuffle: Whether to shuffle the dataset before each epoch
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_token_size = batch_token_size
        self.batch_word_size = batch_word_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Get sequence lengths for each sample
        self.seq_lengths = [len(sample['ground_truth_tokens']) for sample in dataset.samples]
        
        # Get word counts for each sample
        self.word_counts = [sample.get('word_count', 0) for sample in dataset.samples]
        
        # Create indices list
        self.indices = list(range(len(dataset)))
        
        # Calculate workload per worker in distributed setting
        self.num_samples = len(self.indices) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        
        if self.batch_word_size:
            logging.info(f"TokenBucketBatchSampler: {len(dataset)} samples, target {batch_word_size} words per batch")
        else:
            logging.info(f"TokenBucketBatchSampler: {len(dataset)} samples, target {batch_token_size} tokens per batch")
        logging.info(f"Worker {rank}/{num_replicas} will process {self.num_samples} samples")
    
    def __iter__(self):
        # Deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
            
        # Ensure all workers process the same number of samples
        indices = indices[:self.total_size]
        
        # Subsample for this worker
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        # Create batches based on token count or word count
        batches = []
        current_batch = []
        current_token_count = 0
        current_word_count = 0
        
        for idx in indices:
            token_length = self.seq_lengths[idx]
            word_count = self.word_counts[idx] if self.word_counts else 0
            
            # Decide whether to use word-based or token-based batching
            if self.batch_word_size:
                # Word-based batching
                batch_full = (current_word_count + word_count > self.batch_word_size and 
                              len(current_batch) >= self.min_batch_size) or len(current_batch) >= self.max_batch_size
            else:
                # Token-based batching
                batch_full = (current_token_count + token_length > self.batch_token_size and 
                              len(current_batch) >= self.min_batch_size) or len(current_batch) >= self.max_batch_size
            
            if batch_full:
                # Current batch is full, yield it and start a new one
                batches.append(current_batch)
                current_batch = [idx]
                current_token_count = token_length
                current_word_count = word_count
            else:
                # Add this sample to the current batch
                current_batch.append(idx)
                current_token_count += token_length
                current_word_count += word_count
        
        # Add the last batch if not empty and we're not dropping it
        if current_batch and (not self.drop_last or len(current_batch) >= self.min_batch_size):
            batches.append(current_batch)
        
        # Log batch info
        if self.batch_word_size:
            logging.info(f"Worker {self.rank}: Created {len(batches)} batches with target {self.batch_word_size} words per batch")
        else:
            logging.info(f"Worker {self.rank}: Created {len(batches)} batches with target {self.batch_token_size} tokens per batch")
        
        # Log batch size distribution for debugging
        batch_sizes = [len(batch) for batch in batches]
        token_counts = [sum(self.seq_lengths[idx] for idx in batch) for batch in batches]
        
        if self.word_counts:
            word_counts = [sum(self.word_counts[idx] for idx in batch) for batch in batches]
            if word_counts:
                logging.info(f"Word counts - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")
        
        if batch_sizes:
            logging.info(f"Batch sizes - Min: {min(batch_sizes)}, Max: {max(batch_sizes)}, Avg: {sum(batch_sizes)/len(batch_sizes):.1f}")
            logging.info(f"Token counts - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)/len(token_counts):.1f}")
        
        return iter(batches)
    
    def __len__(self):
        # This is an approximation since actual batch count depends on sequence lengths
        if self.drop_last:
            return self.num_samples // self.max_batch_size
        else:
            return (self.num_samples + self.max_batch_size - 1) // self.max_batch_size
    
    def set_epoch(self, epoch):
        """Set the epoch for this sampler to ensure proper shuffling."""
        self.epoch = epoch
        # Update the random generator's seed based on the epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.g = g


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Find max sequence length in batch
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors with -100 padding (ignored by cross_entropy)
    gt_tokens = torch.full((batch_size, max_len), -100, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    
    texts = []
    prompt_texts = []
    audio_paths = []
    audio_ids = []
    word_counts = []
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        gt_tokens[i, :seq_len] = item['ground_truth_tokens']
        seq_lens[i] = seq_len
        
        texts.append(item['text'])
        prompt_texts.append(item['prompt_text'])
        audio_paths.append(item['audio_path'])
        audio_ids.append(item['audio_id'])
        word_counts.append(item.get('word_count', 0))
    
    return {
        'texts': texts,
        'prompt_texts': prompt_texts,
        'ground_truth_tokens': gt_tokens,
        'seq_lens': seq_lens,
        'audio_paths': audio_paths,
        'audio_ids': audio_ids,
        'word_counts': word_counts  # Include word counts in the batch
    }


class EndToEndCFGTrainer:
    """
    End-to-end trainer for CFG scale predictor.
    Computes conditional/unconditional logits during training.
    """
    
    def __init__(self, cfg_predictor: CFGScalePredictor, cosyvoice_model: CosyVoice, optimizer, scheduler, device, args):
        self.cfg_predictor = cfg_predictor
        self.cosyvoice = cosyvoice_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.llm_model = self.cosyvoice.model.llm
        self.sos_eos = 0
        self.task_id = 1
        self.sampling = 25
        self.speech_token_size = 6561
        
        # Initialize tensorboard
        if dist.get_rank() == 0:
            self.writer = SummaryWriter(args.tensorboard_dir)
        else:
            self.writer = None
        
        self.step = 0
        self.epoch = 0
        
        # Freeze CosyVoice parameters - we only train the CFG predictor
        for param in self.cosyvoice.model.llm.parameters():
            param.requires_grad = False
        for param in self.cosyvoice.model.hift.parameters():
            param.requires_grad = False
        for param in self.cosyvoice.model.flow.parameters():
            param.requires_grad = False
            
        logging.info("CosyVoice model frozen, only training CFG predictor")
    
    def _compute_first_step_logits(self, text, prompt_text, audio_path):
        """
        Compute logits for the first step when no previous token exists.
        """
        # This is a placeholder - you need to implement based on your CosyVoice model
        # The key is to run the first step of inference and get conditional/unconditional logits
        
        device = self.device
        vocab_size = self.cosyvoice.model.llm.speech_token_size + 1
        
        # Load reference audio if needed
        reference_audio = None
        if audio_path and os.path.exists(audio_path):
            try:
                reference_audio, _ = load_wav(audio_path, 22050)
                reference_audio = reference_audio.to(device)
            except:
                pass
        
        try:
            # Run the first step of CosyVoice inference to get logits
            # You'll need to adapt this to your specific CosyVoice implementation
            
            # Example approach:
            # 1. Prepare text, prompt, and audio embeddings
            # 2. Run the LLM's forward pass for conditional and unconditional paths
            # 3. Extract logits before CFG is applied
            
            # For now, return dummy logits as placeholder
            cond_logit = torch.randn(vocab_size, device=device)
            uncond_logit = torch.randn(vocab_size, device=device)
            
            return cond_logit, uncond_logit
            
        except Exception as e:
            logging.error(f"Error computing first step logits: {e}")
            # Return zero logits as fallback
            return torch.zeros(vocab_size, device=device), torch.zeros(vocab_size, device=device)
    
    def _compute_next_step_logits(self, text, prompt_text, audio_path, previous_token, timestep):
        """
        Compute logits for next step using the previous guided token as input.
        This is crucial to match inference behavior where the guided token
        from the current step becomes input for the next conditional/unconditional passes.
        """
        # This is a placeholder - you need to implement based on your CosyVoice model
        # The key is to use the previous guided token to compute new logits
        
        device = self.device
        vocab_size = self.cosyvoice.model.llm.speech_token_size + 1
        
        try:
            # Run CosyVoice inference for this timestep
            # Important: Use the previous guided token as input to both conditional and unconditional passes
            
            # Example approach:
            # 1. Set up LLM with the appropriate context
            # 2. Pass in the previous token as input
            # 3. Run conditional and unconditional forward passes
            # 4. Extract logits
            
            # For now, return dummy logits as placeholder
            cond_logit = torch.randn(vocab_size, device=device)
            uncond_logit = torch.randn(vocab_size, device=device)
            
            return cond_logit, uncond_logit
            
        except Exception as e:
            logging.error(f"Error computing next step logits: {e}")
            # Return zero logits as fallback
            return torch.zeros(vocab_size, device=device), torch.zeros(vocab_size, device=device)
    
    def inference_cosyvoice2(self, text, prompt_text, seq_lens):
        model_input_conditional = self.cosyvoice.frontend.frontend_instruct3_batchfy(text, prompt_text)
        # For conditional inputs
        target_text_tokens = model_input_conditional['text']
        target_text_tokens_len = model_input_conditional['text_len']
        prompt_text_tokens = model_input_conditional['prompt_text']
        prompt_text_tokens_len = model_input_conditional['prompt_text_len']
        target_text_tokens_uncond = target_text_tokens
        prompt_text_tokens_uncond = prompt_text_tokens
        target_text_tokens_len_uncond = target_text_tokens_len
        prompt_text_tokens_len_uncond = prompt_text_tokens_len
        
        # Drop the unconditional target text tokens in 50% of the cases
        if np.random.rand() < 0.5:
            target_text_tokens_uncond = None
            target_text_tokens_len_uncond = None
        # Drop the unconditional prompt text tokens in 50% of the cases, but not simultaneously drop both
        else:
            prompt_text_tokens_uncond = None
            prompt_text_tokens_len_uncond = None
        # For conditional inputs
        final_text_tokens_conditional = torch.concat([prompt_text_tokens, target_text_tokens], dim=1)
        final_text_tokens_len = prompt_text_tokens_len + target_text_tokens_len
        # For unconditional inputs
            
        final_text_tokens_uncond = target_text_tokens_uncond if target_text_tokens_uncond is not None else prompt_text_tokens_uncond
        final_text_tokens_len_uncond = target_text_tokens_len_uncond if target_text_tokens_uncond is not None else prompt_text_tokens_len_uncond
        
        text_cond_emb = self.llm_model.llm.model.model.embed_tokens(final_text_tokens_conditional)
        text_uncond_emb = self.llm_model.llm.model.model.embed_tokens(final_text_tokens_uncond)
        
        # Create an empty embedding tensor with the same batch size as the input
        batch_size = final_text_tokens_conditional.size(0)
        embedding = torch.zeros(batch_size, 0, self.cosyvoice.model.llm.llm_input_size, dtype=text_cond_emb.dtype, device=text_cond_emb.device)
        embedding_uncond = torch.zeros(batch_size, 0, self.cosyvoice.model.llm.llm_input_size, dtype=text_cond_emb.dtype, device=text_cond_emb.device)
        
        sos_eos_emb = self.llm_model.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_model.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
                
        prompt_speech_token_emb = torch.zeros(batch_size, 0, self.cosyvoice.model.llm.llm_input_size, dtype=text_cond_emb.dtype, device=text_cond_emb.device)
        prompt_speech_token_uncondition_emb = torch.zeros(batch_size, 0, self.cosyvoice.model.llm.llm_input_size, dtype=text_cond_emb.dtype, device=text_cond_emb.device)
        
        # Batchify special tokens by expanding them to match batch size
        batch_sos_eos_emb = sos_eos_emb.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_size]
        batch_task_id_emb = task_id_emb.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_size]
        
        # Concatenate all embeddings along sequence dimension (dim=1)
        lm_input = torch.cat([batch_sos_eos_emb, embedding, text_cond_emb, batch_task_id_emb, prompt_speech_token_emb], dim=1)
        
        # Same for unconditional
        lm_input_unconditional = torch.cat([batch_sos_eos_emb, embedding_uncond, text_uncond_emb, batch_task_id_emb, prompt_speech_token_uncondition_emb], dim=1)
        
        # Define token ratio parameters if not already defined
        
        # Calculate min and max token lengths for batched inputs
        # Use element-wise operations to maintain batch dimensions
        min_len = (target_text_tokens_len * 2).int()
        max_len = (target_text_tokens_len * 12).int()
        
        max_len = max_len.max()
        out_tokens = [[] for _ in range(batch_size)]
        out_logits = [[] for _ in range(batch_size)]
        active_samples = torch.ones(batch_size, dtype=torch.bool, device=text_cond_emb.device)
        cache = None
        cache_unconditional = None
        cfg_cache = None
        previous_tokens = torch.zeros(batch_size, 1, self.cosyvoice.model.llm.llm_input_size, dtype=text_cond_emb.dtype, device=text_cond_emb.device)
        for i in range(max_len):
            # Break if all samples are done
            if not active_samples.any():
                break
            y_pred, cache = self.llm_model.llm.forward_one_step(lm_input,
                                                            masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                            cache=cache)
            y_pred_unconditional, cache_unconditional = self.llm_model.llm.forward_one_step(lm_input_unconditional,
                                                            masks=torch.tril(torch.ones((1, lm_input_unconditional.shape[1], lm_input_unconditional.shape[1]), device=lm_input_unconditional.device)).to(torch.bool),
                                                            cache=cache_unconditional)
            
            # Calculate the cfg scale using the CFG predictor
            # Clone tensors to avoid memory overlap issues
            cfg_scale, cfg_cache = self.cfg_predictor(y_pred[:, -1].unsqueeze(1), y_pred_unconditional[:, -1].unsqueeze(1), previous_tokens if i == 0 else lm_input, cfg_cache, i)
            # cfg_scale = 1.5
            logits_cond = self.llm_model.llm_decoder(y_pred[:, -1])
            logits_uncond = self.llm_model.llm_decoder(y_pred_unconditional[:, -1])
            logits_guided = logits_uncond + torch.mul(cfg_scale, (logits_cond - logits_uncond))
            # logp_conditional = logits_cond.log_softmax(dim=-1)
            # logp_unconditional = logits_uncond.log_softmax(dim=-1)
            logp_guided = logits_guided.log_softmax(dim=-1)
            # logp_guided = logp_unconditional + cfg_scale * (logp_conditional - logp_unconditional)
            
            # Process each sample in the batch separately
            for b in range(batch_size):
                if not active_samples[b]:
                    continue
                # Apply sampling for each sample with its own min_len
                sample_min_len = min_len[b].item()
                sample_logp = logp_guided[b] if batch_size > 1 else logp_guided.squeeze(0)
                sample_tokens = out_tokens[b] if isinstance(out_tokens, list) and len(out_tokens) > 0 and isinstance(out_tokens[0], list) else out_tokens
                
                top_id = self.llm_model.sampling_ids(
                    sample_logp, 
                    sample_tokens, 
                    self.sampling, 
                    ignore_eos=True if i < sample_min_len else False
                ).item()
                
                if top_id == self.speech_token_size:
                    active_samples[b] = False
                # if top_id > self.speech_token_size:
                #     continue
                
                out_tokens[b].append(top_id)
                out_logits[b].append(logits_guided[b])
            
            
                
            # Gather the latest token from each sample in the batch
            # We need to handle the case where some samples have no tokens yet
            latest_tokens = []
            for b in range(batch_size):
                try:
                    latest_tokens.append([out_tokens[b][-1]])
                except IndexError:
                    print(out_tokens)
                    print(len(out_tokens))
                    print([len(tokens)] for tokens in out_tokens)

            
            # Stack the latest tokens into a tensor for the next iteration
            top_ids = torch.tensor(latest_tokens, device=text_cond_emb.device)

            lm_input = self.llm_model.speech_embedding.weight[top_ids]
            lm_input_unconditional = lm_input
        
        # Convert list of logits to a tensor for further processing (e.g., CE loss calculation)
        if out_logits:
            # First, find the longest sequence among all logits
            max_gen_len = i
            
            # Also consider target sequence length for padding
            target_seq_len = seq_lens.max().item()
            
            # Use the longer of the two lengths to ensure sufficient padding
            max_len = max(max_gen_len, target_seq_len)
            
            # Get vocabulary size and device info
            vocab_size = out_logits[0][0].shape[-1] if out_logits[0] else logits_guided.shape[-1]
            device = logits_guided.device
            dtype = logits_guided.dtype
            
            # First, convert lists of logits to tensors
            stacked_logits = []
            for sample_logits in out_logits:
                if len(sample_logits) > 0:
                    # Stack this sample's logits
                    stacked_logits.append(torch.stack(sample_logits, dim=0))  # [seq_len, vocab_size]
                else:
                    # For empty samples, create a placeholder tensor with 0 sequence length
                    stacked_logits.append(torch.zeros(0, vocab_size, device=device, dtype=dtype))
            
            # Pad along sequence dimension (batch_first=False makes sequence dimension first)
            # Then permute to get [batch_size, max_len, vocab_size]
            padded_tensor = pad_sequence(
                stacked_logits, 
                batch_first=True,  # Get [batch, seq_len, vocab] directly
                padding_value=0.0  # For logits, we still use 0 as padding value
            )
            
            # If the padded tensor isn't long enough (for target sequence lengths), pad more
            if padded_tensor.shape[1] < max_len:
                padding_size = (0, 0, 0, max_len - padded_tensor.shape[1])  # Pad the sequence dimension
                padded_tensor = F.pad(padded_tensor, padding_size, "constant", 0)
                
            # Use the padded tensor as our result
            out_logits_tensor = padded_tensor  # [batch_size, max_len, vocab_size]
            return out_tokens, out_logits_tensor
        else:
            return out_tokens, None
    
    
    def train_batch(self, batch):
        """
        Train on a single batch.
        
        Note: Currently processes the first example in the batch as a demonstration.
        Full batch processing would require extending the autoregressive loop to handle
        multiple sequences simultaneously or processing each sequence in the batch independently.
        """
        
        batch_size = len(batch['texts'])
        loss = 0
        
        text = batch['texts']
        prompt_text = batch['prompt_texts']
        audio_path = batch['audio_paths']
        target_tokens = batch['ground_truth_tokens']
        seq_lens = batch['seq_lens']  # Keep as tensor for batched operations
        
        # Move tensors to device first
        target_tokens = target_tokens.to(self.device)
        seq_lens = seq_lens.to(self.device)
        
        # Now we have a batch of sequences, each with its own length
        # We'll handle each sequence in the batch separately when needed
        
        # Initialize storage for guided tokens and logits
        conditional_logits = []
        unconditional_logits = []
        guided_tokens = []
        
        # Initialize model states and inputs
        # This will depend on your CosyVoice model structure
        device = self.device
        # self.llm_model = self.cosyvoice.model.llm
        
        # Prepare initial inputs (text, prompts, etc.)
        # Note: You'll need to adapt this based on your CosyVoice model
        # This is just a placeholder framework
            # Prepare text inputs, embeddings, etc.
            # text_emb, prompt_emb = self._prepare_inputs(text, prompt_text, audio_path)
        out_tokens, out_logits_tensor = self.inference_cosyvoice2(text, prompt_text, seq_lens)
        
        # Handle cases where the model doesn't generate any logits
        
        # Create a mask to compute loss only on valid positions
        # This ensures we only compute loss where:
        # 1. We have actual target tokens (not -100 padding)
        # 2. We don't exceed each sample's target length
        
        batch_size = target_tokens.size(0)
        max_seq_len = target_tokens.size(1)
        
        # # Create a mask to properly handle sequence lengths and padding
        # valid_mask = torch.arange(max_seq_len, device=self.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        # # Create target tensor padded with -100 for proper loss masking
        # padded_targets = torch.full_like(target_tokens, -100)
        # padded_targets[valid_mask] = target_tokens[valid_mask]
        
        
        # Ensure out_logits_tensor is trimmed to target length if needed
        if out_logits_tensor.size(1) > max_seq_len:
            out_logits_tensor = out_logits_tensor[:, :max_seq_len, :]
        
        # Compute the cross-entropy loss with masked targets
        
        loss = F.cross_entropy(
            out_logits_tensor.reshape(-1, out_logits_tensor.shape[-1]),  # [batch*seq_len, vocab_size]
            target_tokens.reshape(-1),                                 # [batch*seq_len]
            ignore_index=-100,
            reduction='sum'
        )
        
        loss = loss / batch_size  # Average loss over the batch
        # loss = out_logits_tensor.sum()
        return loss
    
    def train_one_epoch(self, train_loader, cv_loader):
        """Train for one epoch."""
        self.cfg_predictor.train()
        total_loss = 0
        num_batches = 0
        total_tokens = 0
        total_words = 0
        
        logging.info(f"GPU {dist.get_rank()}: Starting epoch with {len(train_loader)} batches")
        
        for batch_idx, batch in enumerate(train_loader):
            # Get batch size and token count for this batch
            batch_size = batch['ground_truth_tokens'].size(0)
            token_count = batch['seq_lens'].sum().item()
            word_count = sum(batch['word_counts'])
            total_tokens += token_count
            total_words += word_count

            # logging.info(f"Epoch: {self.epoch} -- GPU {dist.get_rank()}: Processing batch {batch_idx+1}/{len(train_loader)}, "
                        #  f"size: {batch_size} samples, {token_count} tokens, {word_count} words")
            
            # Move batch to device
            batch['ground_truth_tokens'] = batch['ground_truth_tokens'].to(self.device)
            batch['seq_lens'] = batch['seq_lens'].to(self.device)
            
            # Forward pass
            loss = self.train_batch(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()


            if dist.get_rank() == 0:
            # Before clipping, check the grad
                total_norm = 0.0
                for p in self.cfg_predictor.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"[step {batch_idx}] Global grad norm: {total_norm:.4f}")

            # Gradient clipping
            clip_grad = torch.nn.utils.clip_grad_norm_(self.cfg_predictor.parameters(), max_norm=15.0)

            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if self.step % self.args.log_interval == 0 and dist.get_rank() == 0:
                avg_loss = total_loss / max(1, num_batches)
                lr = self.scheduler.get_last_lr()[0]
                avg_tokens_per_batch = total_tokens / max(1, num_batches)
                avg_words_per_batch = total_words / max(1, num_batches)
                
                logging.info(f"Epoch {self.epoch}, Step {self.step}, Loss: {loss.item():.4f}, "
                           f"Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}, "
                           f"Avg tokens/batch: {avg_tokens_per_batch:.1f}, "
                           f"Avg words/batch: {avg_words_per_batch:.1f}")
                
                if self.writer:
                    self.writer.add_scalar('train/loss', loss.item(), self.step)
                    self.writer.add_scalar('train/avg_loss', avg_loss, self.step)
                    self.writer.add_scalar('train/learning_rate', lr, self.step)
                    self.writer.add_scalar('train/tokens_per_batch', avg_tokens_per_batch, self.step)
                    self.writer.add_scalar('train/words_per_batch', avg_words_per_batch, self.step)
            
            # Validation
            if self.step > 0 and self.step % self.args.eval_interval == 0:
                val_loss = self.validate(cv_loader)
                if dist.get_rank() == 0 and self.writer:
                    self.writer.add_scalar('valid/loss', val_loss, self.step)
                
                # Save checkpoint
                if dist.get_rank() == 0:
                    self.save_checkpoint()
            
            self.step += 1
        
        avg_tokens_per_batch = total_tokens / max(1, num_batches)
        avg_words_per_batch = total_words / max(1, num_batches)
        logging.info(f"Epoch completed with avg {avg_tokens_per_batch:.1f} tokens and {avg_words_per_batch:.1f} words per batch")
        return total_loss / max(1, num_batches)
    
    def validate(self, cv_loader):
        """Validation loop."""
        self.cfg_predictor.eval()
        total_loss = 0
        num_batches = 0
        total_tokens = 0
        total_words = 0
        
        with torch.no_grad():
            for batch in cv_loader:
                batch_size = batch['ground_truth_tokens'].size(0)
                token_count = batch['seq_lens'].sum().item()
                word_count = sum(batch['word_counts'])
                total_tokens += token_count
                total_words += word_count
                
                batch['ground_truth_tokens'] = batch['ground_truth_tokens'].to(self.device)
                batch['seq_lens'] = batch['seq_lens'].to(self.device)
                
                loss = self.train_batch(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_tokens_per_batch = total_tokens / max(1, num_batches)
        avg_words_per_batch = total_words / max(1, num_batches)
        logging.info(f"Validation Loss: {avg_loss:.4f}, Avg tokens/batch: {avg_tokens_per_batch:.1f}, Avg words/batch: {avg_words_per_batch:.1f}")
        
        self.cfg_predictor.train()
        return avg_loss
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        save_dict = {
            'cfg_predictor': self.cfg_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'args': self.args
        }
        
        checkpoint_path = os.path.join(self.args.model_dir, f'cfg_predictor_e2e_{self.step}.pt')
        torch.save(save_dict, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint and resume training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
            
        try:
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Check if checkpoint has all required keys
            required_keys = ['cfg_predictor', 'optimizer', 'scheduler', 'step', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                logging.error(f"Checkpoint is missing required keys: {missing_keys}")
                return False
            
            # Load model state
            self.cfg_predictor.load_state_dict(checkpoint['cfg_predictor'])
            
            # Load optimizer and scheduler states
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Load training state
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            
            # Move optimizer states to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            # Ensure args are consistent (we keep the current args but log differences)
            if 'args' in checkpoint:
                saved_args = checkpoint['args']
                current_args = self.args
                # Log any important differences
                important_params = ['learning_rate', 'batch_size', 'batch_token_size', 'max_seq_len']
                for param in important_params:
                    if hasattr(saved_args, param) and hasattr(current_args, param):
                        saved_val = getattr(saved_args, param)
                        current_val = getattr(current_args, param)
                        if saved_val != current_val:
                            logging.warning(f"Parameter '{param}' differs: checkpoint={saved_val}, current={current_val}")
            
            logging.info(f"Successfully loaded checkpoint from epoch {self.epoch}, step {self.step}")
            return True
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_args():
    parser = argparse.ArgumentParser(description='End-to-end CFG Scale Predictor Training')
    
    # Data arguments
    # parser.add_argument('--train_data', required=True, help='Training data file (ground truth tokens)')
    # parser.add_argument('--cv_data', required=True, help='Cross-validation data file')
    parser.add_argument('--model_dir', default="./cfg_predictor_checkpoints", help='Model save directory')
    # parser.add_argument('--cosyvoice_model', required=True, help='CosyVoice model directory')
    parser.add_argument('--tensorboard_dir', default='tensorboard', help='Tensorboard log dir')
    
    # Model arguments
    parser.add_argument('--hidden_dim', default=896, type=int, help='hidden dimention of LLM')
    parser.add_argument('--d_model', default=896, type=int, help='Model dimension')
    parser.add_argument('--nhead', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--num_layers', default=4, type=int, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', default=1536, type=int, help='FFN dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--max_seq_len', default=500, type=int, help='Maximum sequence length')
    parser.add_argument('--cfg_scale_min', default=1.0, type=float, help='Minimum CFG scale')
    parser.add_argument('--cfg_scale_max', default=5.0, type=float, help='Maximum CFG scale')
    
    # Training arguments
    parser.add_argument('--batch_size', default=32, type=int, help='Max samples per batch when using dynamic batching')
    parser.add_argument('--batch_token_size', default=1400, type=int, help='Target tokens per batch for dynamic batching')
    parser.add_argument('--batch_word_size', default=230, type=int, help='Target words per batch for dynamic batching (overrides batch_token_size if specified)')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay')
    parser.add_argument('--warmup_steps', default=1600, type=int, help='Warmup steps')
    parser.add_argument('--log_interval', default=1, type=int, help='Log interval')
    parser.add_argument('--eval_interval', default=400, type=int, help='Evaluation interval')
    
    # Distributed training
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data workers')
    
    # Resume training
    parser.add_argument('--checkpoint', help='Specific checkpoint file to resume from')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in model_dir')
    parser.add_argument('--list-checkpoints', action='store_true', help='List all available checkpoints and exit')
    parser.add_argument('--train_engine', default="torch_ddp")
    
    # Analysis options
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze token length distribution without training')
    parser.add_argument('--train_data', default="data/train.pt", help='Data file to analyze or train on')
    parser.add_argument('--cv_data', default="data/eval.pt", help='Cross-validation data file')


    return parser.parse_args()


def analyze_token_length_distribution(data_file, max_seq_len=None):
    """
    Standalone function to analyze and visualize the distribution of token lengths
    in a dataset without creating a dataset instance.
    
    Args:
        data_file: Path to the PyTorch data file containing samples
        max_seq_len: Optional maximum sequence length filter
    """
    # Load samples from data file
    samples = torch.load(data_file)
    
    # Filter by max_seq_len if provided
    if max_seq_len is not None:
        original_count = len(samples)
        samples = [
            sample for sample in samples 
            if len(sample['ground_truth_tokens']) <= max_seq_len
        ]
        filtered_count = len(samples)
        logging.info(f"Filtered {original_count - filtered_count} samples exceeding max_seq_len {max_seq_len}")
    
    if not samples:
        logging.info("No samples to analyze.")
        return
    
    # Get all lengths
    lengths = [len(sample['ground_truth_tokens']) for sample in samples]
    
    # Calculate statistics
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    median_len = sorted(lengths)[len(lengths) // 2]
    
    # Additional statistics
    p25 = sorted(lengths)[len(lengths) // 4]  # 25th percentile
    p75 = sorted(lengths)[3 * len(lengths) // 4]  # 75th percentile
    p90 = sorted(lengths)[9 * len(lengths) // 10]  # 90th percentile
    
    # Create a histogram of lengths
    hist_bins = 20  # More bins for standalone visualization
    bin_width = (max_len - min_len) / hist_bins if max_len > min_len else 1
    bins = {}
    
    for length in lengths:
        bin_idx = min(hist_bins - 1, int((length - min_len) / bin_width)) if bin_width > 0 else 0
        bin_start = min_len + bin_idx * bin_width
        bin_end = min_len + (bin_idx + 1) * bin_width
        bin_key = f"{int(bin_start)}-{int(bin_end)}"
        
        if bin_key in bins:
            bins[bin_key] += 1
        else:
            bins[bin_key] = 1
    
    # Log the results with more detailed statistics
    logging.info(f"\n{'='*50}")
    logging.info(f"GROUND TRUTH TOKEN LENGTH ANALYSIS ({len(samples)} samples)")
    logging.info(f"{'='*50}")
    logging.info(f"  Min: {min_len}, Max: {max_len}")
    logging.info(f"  Mean: {avg_len:.2f}, Median: {median_len}")
    logging.info(f"  25th percentile: {p25}, 75th percentile: {p75}, 90th percentile: {p90}")
    logging.info(f"\nDistribution by length ranges:")
    
    max_count = max(bins.values())
    scale_factor = 40 / max_count  # Scale to fit in log width
    
    for bin_key, count in sorted(bins.items(), key=lambda x: int(x[0].split('-')[0])):
        percentage = (count / len(lengths)) * 100
        bar = "█" * int(count * scale_factor)
        logging.info(f"  {bin_key:>10}: {count:>5} samples ({percentage:>5.1f}%) {bar}")
    
    logging.info(f"{'='*50}\n")
    
    return {
        "min": min_len,
        "max": max_len,
        "mean": avg_len,
        "median": median_len,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "histogram": bins
    }


def list_checkpoints(model_dir):
    """List available checkpoints in the model directory with their steps and creation dates.
    
    Args:
        model_dir (str): Directory containing checkpoint files
        
    Returns:
        List[Tuple[str, int, datetime]]: List of (filename, step, creation_date) tuples
    """
    import os
    import glob
    from datetime import datetime
    
    checkpoints = glob.glob(os.path.join(model_dir, 'cfg_predictor_e2e_*.pt'))
    result = []
    
    for checkpoint in checkpoints:
        try:
            # Extract step from filename
            step = int(os.path.basename(checkpoint).split('_')[-1].split('.')[0])
            # Get file creation time
            creation_time = datetime.fromtimestamp(os.path.getctime(checkpoint))
            result.append((checkpoint, step, creation_time))
        except (ValueError, IndexError):
            continue
    
    # Sort by step (ascending)
    result.sort(key=lambda x: x[1])
    return result


def main():
    args = get_args()
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Handle list-checkpoints option before initializing distributed training
    if args.list_checkpoints:
        checkpoints = list_checkpoints(args.model_dir)
        if not checkpoints:
            print(f"No checkpoints found in {args.model_dir}")
        else:
            print(f"\nAvailable checkpoints in {args.model_dir}:")
            print(f"{'Filename':60} {'Step':10} {'Creation Date'}")
            print("-" * 100)
            for filename, step, creation_time in checkpoints:
                print(f"{os.path.basename(filename):60} {step:<10} {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nTo resume from a specific checkpoint, use: --checkpoint <checkpoint_path>")
            print(f"To resume from the latest checkpoint, use: --resume\n")
        return
    
    args.train_engine = "torch_ddp" 
    world_size, local_rank, rank = init_distributed(args)
    device = torch.device("cuda", local_rank)
    # If analyze_only is set, just run the analysis and exit
    # if args.analyze_only:
    #     logging.info(f"Analyzing token length distribution in {args.data_file}")
    #     analyze_token_length_distribution(args.data_file, args.max_seq_len)
    #     return
    
    # Multi-GPU setup
    # if torch.cuda.device_count() > 1:
    #     print(f"Setting up for {torch.cuda.device_count()} GPUs")
    # # Enable distributed training environment variables if not already set
    # if "LOCAL_RANK" not in os.environ:
    #     os.environ["LOCAL_RANK"] = "0"
    # if "WORLD_SIZE" not in os.environ:
    #     os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    
    # Initialize distributed training
    # init_distributed(args)
    # dist.init_process_group(backend="nccl")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Multi-GPU handling
    
    
    # Load CosyVoice model
    logging.info("Loading CosyVoice model...")
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
    
    # Create datasets
    train_dataset = EndToEndCFGDataset(args.train_data, args.max_seq_len)
    cv_dataset = EndToEndCFGDataset(args.cv_data, args.max_seq_len)
    
    # Log dataset sizes for debugging
    logging.info(f"GPU {dist.get_rank()}: Total dataset size: {len(train_dataset)} samples")
    
    # Create token-based or word-based batch samplers for dynamic batching
    train_batch_sampler = TokenBucketBatchSampler(
        dataset=train_dataset,
        batch_token_size=args.batch_token_size,
        batch_word_size=args.batch_word_size,  # Use word-based batching if specified
        max_batch_size=args.batch_size,
        min_batch_size=1,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=False
    )
    
    cv_batch_sampler = TokenBucketBatchSampler(
        dataset=cv_dataset,
        batch_token_size=args.batch_token_size,
        batch_word_size=args.batch_word_size,  # Use word-based batching if specified
        max_batch_size=args.batch_size,
        min_batch_size=1,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False
    )
    
    # Create data loaders with the token-based batch samplers
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,  # Use token-based batch sampler
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    cv_loader = DataLoader(
        cv_dataset,
        batch_sampler=cv_batch_sampler,  # Use token-based batch sampler
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create CFG predictor model
    cfg_predictor = CFGScalePredictor(
        hidden_dim=args.hidden_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=1024,
        cfg_scale_range=(args.cfg_scale_min, args.cfg_scale_max)
    )
    cfg_predictor = cfg_predictor.to(device)
    # Move model to device and wrap for distributed training
    cfg_predictor = wrap_cuda_model(args, cfg_predictor)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        cfg_predictor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=args.warmup_steps
    )

    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20000 - args.warmup_steps
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[args.warmup_steps]
    )
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load checkpoint if provided or if resume is requested
    start_epoch = 0
    start_step = 0
    
    # Initialize trainer attributes
    trainer = EndToEndCFGTrainer(cfg_predictor, cosyvoice, optimizer, scheduler, device, args)
    
    # Set step to 0 by default (will be updated if loading checkpoint)
    trainer.step = 0
    trainer.epoch = 0
    
    if args.resume:
        # Find the latest checkpoint in the model directory
        checkpoints = glob.glob(os.path.join(args.model_dir, 'cfg_predictor_e2e_*.pt'))
        if checkpoints:
            # Sort by step number
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))[-1]
            logging.info(f"Found latest checkpoint: {latest_checkpoint}")
            args.checkpoint = latest_checkpoint
        else:
            logging.warning(f"No checkpoints found in {args.model_dir}, starting from scratch")
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        # Use the trainer's load_checkpoint method
        if trainer.load_checkpoint(args.checkpoint):
            logging.info(f"Successfully resumed training from epoch {trainer.epoch}, step {trainer.step}")
        else:
            logging.warning("Failed to load checkpoint, starting from scratch")
    
    # Training loop
    for epoch in range(trainer.epoch, args.num_epochs):
        trainer.epoch = epoch
        logging.info(f"Starting epoch {epoch}")
        
        # Set epoch for proper data shuffling per epoch
        train_loader.batch_sampler.set_epoch(epoch)
        
        avg_loss = trainer.train_one_epoch(train_loader, cv_loader)
        
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch} completed, average loss: {avg_loss:.4f}")
    
    logging.info("End-to-end training completed!")


if __name__ == '__main__':
    main()
