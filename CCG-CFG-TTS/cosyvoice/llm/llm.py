# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy

import numpy as np

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            embedding_unconditional: Optional[torch.Tensor] = None,
            prompt_text_unconditional: Optional[torch.Tensor] = None,
            prompt_text_unconditional_len: Optional[torch.Tensor] = None,
            prompt_speech_token_unconditional: Optional[torch.Tensor] = None,
            prompt_speech_token_len_unconditional: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            
    ) -> Generator[torch.Tensor, None, None]:
        if self.fp16 is True:
            embedding = embedding.half()
            if embedding_unconditional is not None:
                embedding_unconditional = embedding_unconditional.half()
        cfg = False
        if prompt_text_unconditional is not None:
            assert prompt_text_unconditional_len is not None, "prompt_text_uncondition_len must be provided if prompt_text_uncondition is provided"
            cfg = True
        elif prompt_speech_token_unconditional is not None:
            assert prompt_speech_token_len_unconditional is not None, "prompt_speech_token_len_uncondition must be provided if prompt_speech_token_uncondition is provided"
            cfg = True

        if cfg:
            assert cfg_scale > 0, "cfg_scale must be greater than 0 if cfg is True"
            device = text.device
            text_conditional = torch.concat([prompt_text, text], dim=1)
            text_conditional_len = text_len + prompt_text_len
            text_conditional = self.text_embedding(text_conditional)
            
            text_unconditional = torch.concat([prompt_text_unconditional, text], dim=1)
            text_unconditional_len = text_len + prompt_text_unconditional_len
            text_unconditional = self.text_embedding(text_unconditional)

            # 1. encode text
            text_conditional, text_conditional_len = self.encode(text_conditional, text_conditional_len)
            text_unconditional, text_unconditional_len = self.encode(text_unconditional, text_unconditional_len)
            
            # 2. encode embedding
            if embedding.shape[0] != 0:
                embedding = F.normalize(embedding, dim=1)
                embedding = self.spk_embed_affine_layer(embedding)
                embedding = embedding.unsqueeze(dim=1)
            else:
                embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
                
            if embedding_unconditional is not None and embedding_unconditional.shape[0] != 0:
                embedding_unconditional = F.normalize(embedding_unconditional, dim=1)
                embedding_unconditional = self.spk_embed_affine_layer(embedding_unconditional)
                embedding_unconditional = embedding_unconditional.unsqueeze(dim=1)
            else:
                embedding_unconditional = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
                
            # 3. concat llm_input
            sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
            task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
            if prompt_speech_token_len != 0:
                prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
            else:
                prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
            if prompt_speech_token_len_unconditional != 0:
                prompt_speech_token_uncondition_emb = self.speech_embedding(prompt_speech_token_unconditional)
            else:
                prompt_speech_token_uncondition_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
            lm_input_conditional = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
            lm_input_unconditional = torch.concat([sos_eos_emb, embedding_unconditional, text_unconditional, task_id_emb, prompt_speech_token_uncondition_emb], dim=1)
            
            # 4. cal min/max_length
            min_len = int((text_conditional_len - prompt_text_len) * min_token_text_ratio)
            max_len = int((text_conditional_len - prompt_text_len) * max_token_text_ratio)
            
            # 5. Step by step decode
            out_tokens = []
            offset = 0
            att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input_conditional.device), torch.zeros((0, 0, 0, 0), device=lm_input_conditional.device)
            for i in range(max_len):
                y_pred_conditional, att_cache, cnn_cache = self.llm.forward_chunk(lm_input_conditional, offset=offset, required_cache_size=-1,
                                                                    att_cache=att_cache, cnn_cache=cnn_cache,
                                                                    att_mask=torch.tril(torch.ones((1, lm_input_conditional.shape[1], lm_input_conditional.shape[1]),
                                                                                                    device=lm_input_conditional.device)).to(torch.bool))
                y_pred_unconditional, att_cache_unconditional, cnn_cache_unconditional = self.llm.forward_chunk(lm_input_unconditional, offset=offset, required_cache_size=-1,
                                                                    att_cache=att_cache_unconditional, cnn_cache=cnn_cache_unconditional,
                                                                    att_mask=torch.tril(torch.ones((1, lm_input_unconditional.shape[1], lm_input_unconditional.shape[1]),
                                                                                                    device=lm_input_unconditional.device)).to(torch.bool))
                logp_conditional = self.llm_decoder(y_pred_conditional[:, -1]).log_softmax(dim=-1)
                logp_unconditional = self.llm_decoder(y_pred_unconditional[:, -1]).log_softmax(dim=-1)
                guided_logp = logp_unconditional - cfg_scale * (logp_unconditional - logp_conditional) if cfg_scale > 1.0 else logp_conditional if cfg_scale > 1.0 else logp_conditional
                # force continue decode first token
                if i == 0:
                    guided_logp[:, self.speech_token_size] = -float('inf')
                top_ids = self.sampling_ids(guided_logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                offset += lm_input_conditional.size(1)
                lm_input_conditional = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        if not cfg:
            device = text.device
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
            text = self.text_embedding(text)

            # 1. encode text
            text, text_len = self.encode(text, text_len)

            # 2. encode embedding
            if embedding.shape[0] != 0:
                embedding = F.normalize(embedding, dim=1)
                embedding = self.spk_embed_affine_layer(embedding)
                embedding = embedding.unsqueeze(dim=1)
            else:
                embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

            # 3. concat llm_input
            sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
            task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
            if prompt_speech_token_len != 0:
                prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
            else:
                prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
            lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

            # 4. cal min/max_length
            min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
            max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

            # 5. step by step decode
            out_tokens = []
            offset = 0
            att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
            for i in range(max_len):
                y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                    att_cache=att_cache, cnn_cache=cnn_cache,
                                                                    att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                    device=lm_input.device)).to(torch.bool))
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                # force continue decode first token
                if i == 0:
                    logp[:, self.speech_token_size] = -float('inf')
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                offset += lm_input.size(1)
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 25 # 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                break
                # raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            embedding_unconditional: Optional[torch.Tensor] = None,
            prompt_text_unconditional: Optional[torch.Tensor] = None,
            prompt_text_unconditional_len: Optional[torch.Tensor] = None,
            prompt_speech_token_unconditional: Optional[torch.Tensor] = None,
            prompt_speech_token_len_unconditional: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = 1.0,
            cfg_scale_list: Optional[List[float]] = None,
            cfg_filter_topk: Optional[int] = -1,
            cfg_drop_prompt: Optional[bool] = False,
            cfg_drop_target: Optional[bool] = False,
            cfg_rescale: Optional[bool] = 1.0,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        cfg = False
        dual_cfg = False
        if cfg_drop_prompt != cfg_drop_target:
            if prompt_text_unconditional is not None and cfg_drop_prompt:
                cfg = True
                text_unconditional = torch.concat([prompt_text_unconditional, text], dim=1)
                # print("Unconditional Text:", text_unconditional)
                text_unconditional_len = text_len + prompt_text_unconditional_len
                text_unconditional = self.llm.model.model.embed_tokens(text_unconditional)
            if text is not None and cfg_drop_target:
                cfg = True
                text_unconditional = prompt_text
                text_unconditional_len = prompt_text_len
                text_unconditional = self.llm.model.model.embed_tokens(text_unconditional)
        elif cfg_scale != 1.0:
            dual_cfg = True
            text_unconditional_prompt = torch.concat([prompt_text_unconditional, text], dim=1)
            text_unconditional_prompt_len = text_len + prompt_text_unconditional_len
            text_unconditional_prompt = self.llm.model.model.embed_tokens(text_unconditional_prompt)
            
            text_unconditional_target = prompt_text
            text_unconditional_target_len = prompt_text_len
            text_unconditional_target = self.llm.model.model.embed_tokens(text_unconditional_target)
            
        text = torch.concat([prompt_text, text], dim=1)
        # print("Conditional Text:", text)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)


        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
        if cfg:
            embedding_unconditional = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
        if dual_cfg:
            embedding_unconditional_prompt = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
            embedding_unconditional_target = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input_unconditional = None
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        if cfg:
            if prompt_speech_token_len_unconditional != 0:
                prompt_speech_token_uncondition_emb = self.speech_embedding(prompt_speech_token_unconditional)  
            else:
                prompt_speech_token_uncondition_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
            lm_input_unconditional = torch.concat([sos_eos_emb, embedding_unconditional, text_unconditional, task_id_emb, prompt_speech_token_uncondition_emb], dim=1)
        
        if dual_cfg:
            if prompt_speech_token_len_unconditional != 0:
                prompt_speech_token_uncondition_emb_prompt = self.speech_embedding(prompt_speech_token_unconditional)  
            else:
                prompt_speech_token_uncondition_emb_prompt = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
            lm_input_unconditional_prompt = torch.concat([sos_eos_emb, embedding_unconditional_prompt, text_unconditional_prompt, task_id_emb, prompt_speech_token_uncondition_emb_prompt], dim=1)
            
            if prompt_speech_token_len != 0:
                prompt_speech_token_emb_target = self.speech_embedding(prompt_speech_token)  
            else:
                prompt_speech_token_emb_target = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
            lm_input_unconditional_target = torch.concat([sos_eos_emb, embedding_unconditional_target, text_unconditional_target, task_id_emb, prompt_speech_token_emb_target], dim=1)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        cache_unconditional = None
        cache_unconditional_prompt = None
        cache_unconditional_target = None
        if cfg:
            for i in range(max_len):
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                        cache=cache)
                y_pred_unconditional, cache_unconditional = self.llm.forward_one_step(lm_input_unconditional,
                                                        masks=torch.tril(torch.ones((1, lm_input_unconditional.shape[1], lm_input_unconditional.shape[1]), device=lm_input_unconditional.device)).to(torch.bool),
                                                        cache=cache_unconditional)
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                logp_unconditional = self.llm_decoder(y_pred_unconditional[:, -1]).log_softmax(dim=-1)
                # per_token_guide = True
                # if per_token_guide:
                #     cfg_scale = np.random.random() + 1.5
                logp_guided = logp_unconditional + cfg_scale * (logp - logp_unconditional) if cfg_scale > 1.0 else logp if cfg_scale > 1.0 else logp
                if cfg_filter_topk == -1:
                    top_ids = self.sampling_ids(logp_guided.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                else:
                    # Optimized version using PyTorch operations
                    _, topk_idx = torch.topk(logp_guided[0], cfg_filter_topk, dim=-1)
                    masked = torch.full_like(logp_guided, -float('inf'))
                    masked[0].scatter_(0, topk_idx, logp[0].gather(0, topk_idx))
                    if cfg_rescale > 1.0:
                        masked = logp_unconditional + cfg_rescale * (masked - logp_unconditional)
                    top_ids = self.sampling_ids(masked.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                if top_ids > self.speech_token_size:
                    continue
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                lm_input_unconditional = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        elif dual_cfg:
            for i in range(max_len):
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                        cache=cache)
                y_pred_unconditional_prompt, cache_unconditional_prompt = self.llm.forward_one_step(lm_input_unconditional_prompt,
                                                        masks=torch.tril(torch.ones((1, lm_input_unconditional_prompt.shape[1], lm_input_unconditional_prompt.shape[1]), device=lm_input_unconditional_prompt.device)).to(torch.bool),
                                                        cache=cache_unconditional_prompt)
                y_pred_unconditional_target, cache_unconditional_target = self.llm.forward_one_step(lm_input_unconditional_target,
                                                        masks=torch.tril(torch.ones((1, lm_input_unconditional_target.shape[1], lm_input_unconditional_target.shape[1]), device=lm_input_unconditional_target.device)).to(torch.bool),
                                                        cache=cache_unconditional_target)
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                logp_unconditional_prompt = self.llm_decoder(y_pred_unconditional_prompt[:, -1]).log_softmax(dim=-1)
                logp_unconditional_target = self.llm_decoder(y_pred_unconditional_target[:, -1]).log_softmax(dim=-1)
                # per_token_guide = True
                # if per_token_guide:
                #     cfg_scale = np.random.random() + 1.5
                # The original formula over-emphasizes the conditional logp and uses undefined variables.
                # A more balanced approach for dual guidance is to average the unconditional probabilities
                # and apply the standard CFG formula.
                logp_unconditional_avg = (logp_unconditional_prompt + logp_unconditional_target) / 2.0
                logp_guided = logp_unconditional_avg + cfg_scale_list[0] * (logp - logp_unconditional_prompt) + cfg_scale_list[1] * (logp - logp_unconditional_target)
                if cfg_filter_topk == -1:
                    top_ids = self.sampling_ids(logp_guided.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                else:
                    # Optimized version using PyTorch operations
                    _, topk_idx = torch.topk(logp_guided[0], cfg_filter_topk, dim=-1)
                    masked = torch.full_like(logp_guided, -float('inf'))
                    masked[0].scatter_(0, topk_idx, logp[0].gather(0, topk_idx))
                    if cfg_rescale > 1.0:
                        masked = logp_unconditional_avg + cfg_rescale * (masked - logp_unconditional_avg)
                    top_ids = self.sampling_ids(masked.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                if top_ids > self.speech_token_size:
                    continue
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                lm_input_unconditional_prompt = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                lm_input_unconditional_target = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        
        else:
            for i in range(max_len):
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                        cache=cache)
                
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                if top_ids > self.speech_token_size:
                    continue
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
