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
import argparse
import datetime
import logging
import os
import json
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from cosyvoice.models.cfg_predictor import CFGScalePredictor, apply_guided_sampling
from cosyvoice.utils.train_utils import init_distributed, wrap_cuda_model, save_model


class CFGTrainingDataset(Dataset):
    """
    Dataset for training the CFG scale predictor.
    This dataset should contain pre-extracted logits from CosyVoice2 LM
    and corresponding ground truth tokens.
    """
    
    def __init__(self, data_file: str, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len
        self.samples = []
        
        # Load pre-extracted data
        # Expected format: each line is a JSON with:
        # {
        #   "conditional_logits": [...],     # List of logits for each timestep
        #   "unconditional_logits": [...],   # List of logits for each timestep  
        #   "ground_truth_tokens": [...],    # Ground truth token sequence
        #   "text_len": int,                 # Length of text portion
        #   "audio_id": str                  # Sample identifier
        # }
        
        with open(data_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                if len(sample['ground_truth_tokens']) <= max_seq_len:
                    self.samples.append(sample)
        
        logging.info(f"Loaded {len(self.samples)} samples from {data_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        conditional_logits = torch.tensor(sample['conditional_logits'], dtype=torch.float32)
        unconditional_logits = torch.tensor(sample['unconditional_logits'], dtype=torch.float32)
        ground_truth_tokens = torch.tensor(sample['ground_truth_tokens'], dtype=torch.long)
        
        seq_len = len(ground_truth_tokens)
        
        # Generate optimal CFG scales using a heuristic or pre-computed values
        # For now, we'll use a simple approach: try different CFG scales and pick the best
        target_cfg_scales = self.compute_optimal_cfg_scales(
            conditional_logits, unconditional_logits, ground_truth_tokens
        )
        
        return {
            'conditional_logits': conditional_logits,      # [seq_len, vocab_size]
            'unconditional_logits': unconditional_logits,  # [seq_len, vocab_size]
            'ground_truth_tokens': ground_truth_tokens,    # [seq_len]
            'target_cfg_scales': target_cfg_scales,        # [seq_len]
            'seq_len': seq_len,
            'audio_id': sample.get('audio_id', f'sample_{idx}')
        }
    
    def compute_optimal_cfg_scales(self, cond_logits, uncond_logits, gt_tokens):
        """
        Compute optimal CFG scales for each timestep by trying different values
        and selecting the one that gives highest probability for ground truth token.
        """
        seq_len = len(gt_tokens)
        optimal_scales = torch.zeros(seq_len)
        
        # Test different CFG scale values
        cfg_candidates = torch.linspace(0.5, 3.5, 15)  # Test 15 different scales
        
        for t in range(seq_len):
            gt_token = gt_tokens[t]
            best_scale = 1.0
            best_prob = -float('inf')
            
            for cfg_scale in cfg_candidates:
                # Apply guided sampling
                guided_logp = apply_guided_sampling(
                    cond_logits[t:t+1], uncond_logits[t:t+1], cfg_scale.item()
                )
                
                # Get probability of ground truth token
                prob = guided_logp[0, gt_token].item()
                
                if prob > best_prob:
                    best_prob = prob
                    best_scale = cfg_scale.item()
            
            optimal_scales[t] = best_scale
        
        return optimal_scales


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Find max sequence length in batch
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    vocab_size = batch[0]['conditional_logits'].shape[-1]
    
    # Initialize padded tensors
    cond_logits = torch.zeros(batch_size, max_len, vocab_size)
    uncond_logits = torch.zeros(batch_size, max_len, vocab_size)
    gt_tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_scales = torch.zeros(batch_size, max_len)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        cond_logits[i, :seq_len] = item['conditional_logits']
        uncond_logits[i, :seq_len] = item['unconditional_logits']
        gt_tokens[i, :seq_len] = item['ground_truth_tokens']
        target_scales[i, :seq_len] = item['target_cfg_scales']
        seq_lens[i] = seq_len
    
    return {
        'conditional_logits': cond_logits,
        'unconditional_logits': uncond_logits,
        'ground_truth_tokens': gt_tokens,
        'target_cfg_scales': target_scales,
        'seq_lens': seq_lens,
        'audio_ids': [item['audio_id'] for item in batch]
    }


class CFGTrainer:
    """Trainer for the CFG scale predictor."""
    
    def __init__(self, model, optimizer, scheduler, device, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        
        # Initialize tensorboard
        if dist.get_rank() == 0:
            self.writer = SummaryWriter(args.tensorboard_dir)
        else:
            self.writer = None
        
        self.step = 0
        self.epoch = 0
    
    def train_one_epoch(self, train_loader, cv_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            for key in ['conditional_logits', 'unconditional_logits', 'ground_truth_tokens', 'target_cfg_scales']:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass
            outputs = self.model(
                conditional_logits=batch['conditional_logits'],
                unconditional_logits=batch['unconditional_logits'],
                previous_tokens=batch['ground_truth_tokens'],
                target_cfg_scales=batch['target_cfg_scales']
            )
            
            loss = outputs['cfg_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if self.step % self.args.log_interval == 0 and dist.get_rank() == 0:
                avg_loss = total_loss / max(1, num_batches)
                lr = self.scheduler.get_last_lr()[0]
                
                logging.info(f"Epoch {self.epoch}, Step {self.step}, Loss: {loss.item():.4f}, "
                           f"Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}")
                
                if self.writer:
                    self.writer.add_scalar('train/loss', loss.item(), self.step)
                    self.writer.add_scalar('train/avg_loss', avg_loss, self.step)
                    self.writer.add_scalar('train/learning_rate', lr, self.step)
            
            # Validation
            if self.step > 0 and self.step % self.args.eval_interval == 0:
                val_loss = self.validate(cv_loader)
                if dist.get_rank() == 0 and self.writer:
                    self.writer.add_scalar('valid/loss', val_loss, self.step)
                
                # Save checkpoint
                if dist.get_rank() == 0:
                    self.save_checkpoint()
            
            self.step += 1
        
        return total_loss / max(1, num_batches)
    
    def validate(self, cv_loader):
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in cv_loader:
                # Move batch to device
                for key in ['conditional_logits', 'unconditional_logits', 'ground_truth_tokens', 'target_cfg_scales']:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    conditional_logits=batch['conditional_logits'],
                    unconditional_logits=batch['unconditional_logits'],
                    previous_tokens=batch['ground_truth_tokens'],
                    target_cfg_scales=batch['target_cfg_scales']
                )
                
                loss = outputs['cfg_loss']
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        logging.info(f"Validation Loss: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'args': self.args
        }
        
        checkpoint_path = os.path.join(self.args.model_dir, f'cfg_predictor_{self.step}.pt')
        torch.save(save_dict, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Train CFG Scale Predictor')
    
    # Data arguments
    parser.add_argument('--train_data', required=True, help='Training data file')
    parser.add_argument('--cv_data', required=True, help='Cross-validation data file')
    parser.add_argument('--model_dir', required=True, help='Model save directory')
    parser.add_argument('--tensorboard_dir', default='tensorboard', help='Tensorboard log dir')
    
    # Model arguments
    parser.add_argument('--vocab_size', default=4096, type=int, help='Vocabulary size')
    parser.add_argument('--d_model', default=256, type=int, help='Model dimension')
    parser.add_argument('--nhead', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--num_layers', default=4, type=int, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='FFN dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--max_seq_len', default=1024, type=int, help='Maximum sequence length')
    parser.add_argument('--cfg_scale_min', default=0.0, type=float, help='Minimum CFG scale')
    parser.add_argument('--cfg_scale_max', default=4.0, type=float, help='Maximum CFG scale')
    
    # Training arguments
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--warmup_steps', default=4000, type=int, help='Warmup steps')
    parser.add_argument('--log_interval', default=100, type=int, help='Log interval')
    parser.add_argument('--eval_interval', default=1000, type=int, help='Evaluation interval')
    
    # Distributed training
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data workers')
    
    # Resume training
    parser.add_argument('--checkpoint', help='Checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    # Initialize distributed training
    init_distributed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = CFGTrainingDataset(args.train_data, args.max_seq_len)
    cv_dataset = CFGTrainingDataset(args.cv_data, args.max_seq_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    cv_loader = DataLoader(
        cv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = CFGScalePredictor(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        cfg_scale_range=(args.cfg_scale_min, args.cfg_scale_max)
    )
    
    # Move model to device and wrap for distributed training
    model = wrap_cuda_model(args, model)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_steps
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Resumed from checkpoint {args.checkpoint}")
    
    # Create trainer
    trainer = CFGTrainer(model, optimizer, scheduler, device, args)
    trainer.epoch = start_epoch
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        trainer.epoch = epoch
        logging.info(f"Starting epoch {epoch}")
        
        avg_loss = trainer.train_one_epoch(train_loader, cv_loader)
        
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch} completed, average loss: {avg_loss:.4f}")
    
    logging.info("Training completed!")


if __name__ == '__main__':
    main()
