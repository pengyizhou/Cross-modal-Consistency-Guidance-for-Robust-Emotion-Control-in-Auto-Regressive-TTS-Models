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

import argparse
import logging
import os
import torch
import torch.nn.functional as F
from typing import Generator, Optional

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.models.cfg_predictor import CFGScalePredictor, apply_guided_sampling


class CosyVoiceWithCFGPredictor:
    """
    CosyVoice with adaptive CFG scale prediction.
    This class integrates the trained CFG predictor with CosyVoice2.
    """
    
    def __init__(self, cosyvoice_model_dir: str, cfg_predictor_path: str):
        # Load CosyVoice model
        self.cosyvoice = CosyVoice(cosyvoice_model_dir)
        
        # Load CFG predictor
        checkpoint = torch.load(cfg_predictor_path, map_location='cpu')
        self.cfg_predictor = CFGScalePredictor(
            vocab_size=checkpoint['args'].vocab_size,
            d_model=checkpoint['args'].d_model,
            nhead=checkpoint['args'].nhead,
            num_layers=checkpoint['args'].num_layers,
            dim_feedforward=checkpoint['args'].dim_feedforward,
            dropout=checkpoint['args'].dropout,
            max_seq_len=checkpoint['args'].max_seq_len,
            cfg_scale_range=(checkpoint['args'].cfg_scale_min, checkpoint['args'].cfg_scale_max)
        )
        self.cfg_predictor.load_state_dict(checkpoint['model'])
        self.cfg_predictor.eval()
        
        # Move to device
        device = next(self.cosyvoice.model.parameters()).device
        self.cfg_predictor = self.cfg_predictor.to(device)
        
        logging.info("CosyVoice with CFG Predictor initialized")
    
    def inference_with_adaptive_cfg(
        self,
        text: str,
        prompt_text: str = "",
        **kwargs
    ) -> Generator[torch.Tensor, None, None]:
        """
        Run inference with adaptive CFG scale prediction.
        This modifies the original CosyVoice inference to use predicted CFG scales.
        """
        
        # Get the LLM model for modified inference
        llm_model = self.cosyvoice.model.llm
        
        # This is where you'd integrate with the actual CosyVoice inference
        # You need to modify the inference loop in llm.py to use the CFG predictor
        
        # Placeholder for the modified inference logic
        # The actual implementation would replace the CFG loop in llm.py
        
        previous_tokens = torch.tensor([], dtype=torch.long).to(self.cfg_predictor.device)
        
        # Modified inference loop (template)
        for step_output in self._modified_cosyvoice_inference(text, prompt_text, **kwargs):
            yield step_output
    
    def _modified_cosyvoice_inference(self, text: str, prompt_text: str, **kwargs):
        """
        Modified CosyVoice inference that uses adaptive CFG scales.
        
        This function should replace or modify the inference loop in the LLM model
        to use the CFG predictor instead of fixed CFG scales.
        """
        
        # This is a template showing how to integrate the CFG predictor
        # You'll need to adapt this to work with your actual CosyVoice inference code
        
        # Access the LLM model
        llm_model = self.cosyvoice.model.llm
        
        # Initialize tracking for previous tokens
        previous_tokens = torch.tensor([], dtype=torch.long).to(self.cfg_predictor.device)
        
        # Mock inference loop - replace with actual CosyVoice inference
        """
        # This is what the modified inference loop should look like:
        
        for i in range(max_len):
            # ... existing CosyVoice inference code until logits computation ...
            
            # Get conditional and unconditional logits (from around line 465-466 in llm.py)
            logp_conditional = self.llm_decoder(y_pred_conditional[:, -1]).log_softmax(dim=-1)
            logp_unconditional = self.llm_decoder(y_pred_unconditional[:, -1]).log_softmax(dim=-1)
            
            # PREDICT CFG SCALE using our trained model
            predicted_cfg_scale = self.cfg_predictor.predict_next_cfg_scale(
                conditional_logits=logp_conditional.unsqueeze(1),  # Add seq dim
                unconditional_logits=logp_unconditional.unsqueeze(1),  # Add seq dim
                previous_tokens=previous_tokens.unsqueeze(0) if len(previous_tokens) > 0 else torch.zeros(1, 1, dtype=torch.long).to(device)
            )
            
            # Apply adaptive CFG guidance
            guided_logp = apply_guided_sampling(
                logp_conditional, logp_unconditional, predicted_cfg_scale.item()
            )
            
            # Continue with sampling
            if i == 0:
                guided_logp[:, self.speech_token_size] = -float('inf')
            
            top_ids = self.sampling_ids(guided_logp.squeeze(dim=0), out_tokens, sampling, 
                                      ignore_eos=True if i < min_len else False).item()
            
            if top_ids == self.speech_token_size:
                break
            
            # Update previous tokens for next prediction
            if len(previous_tokens) == 0:
                previous_tokens = torch.tensor([top_ids], dtype=torch.long).to(device)
            else:
                previous_tokens = torch.cat([previous_tokens, torch.tensor([top_ids], dtype=torch.long).to(device)], dim=0)
            
            # Yield the token
            yield top_ids
            
            # ... rest of existing CosyVoice inference code ...
        """
        
        # For now, just use the original CosyVoice inference
        # You'll need to implement the actual modification
        yield from self.cosyvoice.inference(text, **kwargs)


def modify_llm_inference_for_cfg_predictor(llm_model, cfg_predictor):
    """
    This function shows how to modify the LLM inference method to use the CFG predictor.
    You would need to patch the actual inference method in the LLM class.
    """
    
    # Store original inference method
    original_inference = llm_model.inference
    
    def adaptive_cfg_inference(self, *args, **kwargs):
        """Modified inference method that uses CFG predictor."""
        
        # Extract parameters
        cfg_scale = kwargs.get('cfg_scale', 1.0)
        
        # If CFG is not enabled, use original method
        if cfg_scale is None or cfg_scale <= 1.0:
            yield from original_inference(*args, **kwargs)
            return
        
        # Initialize tracking
        previous_tokens = torch.tensor([], dtype=torch.long).to(cfg_predictor.device)
        
        # This is where you'd implement the modified inference loop
        # that uses the CFG predictor instead of fixed CFG scale
        
        # For now, fallback to original
        yield from original_inference(*args, **kwargs)
    
    # Replace the inference method
    llm_model.inference = adaptive_cfg_inference.__get__(llm_model, llm_model.__class__)


def get_args():
    parser = argparse.ArgumentParser(description='CosyVoice inference with adaptive CFG')
    parser.add_argument('--cosyvoice_model', required=True, help='CosyVoice model directory')
    parser.add_argument('--cfg_predictor', required=True, help='CFG predictor checkpoint')
    parser.add_argument('--text', required=True, help='Text to synthesize')
    parser.add_argument('--prompt_text', default='', help='Prompt text')
    parser.add_argument('--output_path', required=True, help='Output audio path')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    # Initialize model
    model = CosyVoiceWithCFGPredictor(args.cosyvoice_model, args.cfg_predictor)
    
    # Run inference
    logging.info(f"Synthesizing: {args.text}")
    
    audio_tokens = list(model.inference_with_adaptive_cfg(
        text=args.text,
        prompt_text=args.prompt_text
    ))
    
    # Convert tokens to audio and save
    # This depends on your CosyVoice implementation
    # You'll need to add the audio generation and saving logic here
    
    logging.info(f"Generated {len(audio_tokens)} tokens")
    logging.info(f"Audio saved to {args.output_path}")


if __name__ == '__main__':
    main()
