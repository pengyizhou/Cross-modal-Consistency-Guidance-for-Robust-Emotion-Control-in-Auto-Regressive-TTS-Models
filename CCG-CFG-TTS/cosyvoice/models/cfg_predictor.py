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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))  # Register as a non-trainable parameter
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CFGScalePredictor(nn.Module):
    """
    Lightweight transformer decoder that predicts token-level CFG scale values.
    Takes conditional and unconditional LLM hidden states (before output layer) and predicts
    optimal CFG scale for each timestep.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        cfg_scale_range: tuple = (1.0, 5.0),
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim
        self.cfg_scale_range = cfg_scale_range
        
        # Input projection layers
        # Project conditional and unconditional hidden states to d_model
        self.linear_proj = nn.Linear(self.hidden_dim * 2, self.d_model)
        # self.unconditional_proj = nn.Linear(self.hidden_dim, self.d_model // 2)
        # self.token_proj = nn.Linear(hidden_dim, d_model)
        
        # Token embedding for previous tokens (for autoregressive prediction)
        # self.token_embedding = nn.Embedding(8192, d_model)  # Using standard 8K vocab size
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, 0.1, max_seq_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output head to predict CFG scale (single continuous value)
        self.cfg_scale_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, then scale to cfg_scale_range
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(
        self,
        conditional_hidden: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
        unconditional_hidden: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
        previous_tokens: torch.Tensor,  # [batch_size, seq_len] - previous predictions
        cache: Optional[Dict] = None,  # Cache for previous computations
        timestep: int = 0,  # Current timestep
    ) -> Dict[str, torch.Tensor]:

        output = self.forward_incremental(conditional_hidden, unconditional_hidden, previous_tokens, cache, timestep)
        # batch_size, seq_len = conditional_hidden.shape[:2]
        
        # # Project hidden states to feature space
        # cond_features = self.conditional_proj(conditional_hidden)  # [B, L, d_model//2]
        # uncond_features = self.unconditional_proj(unconditional_hidden)  # [B, L, d_model//2]
        
        # # Concatenate conditional and unconditional features
        # hidden_features = torch.cat([cond_features, uncond_features], dim=-1)  # [B, L, d_model]
        
        # # Get token embeddings for previous tokens
        
        # token_emb = previous_tokens  # [B, L, d_model]        
        # # Combine hidden state features and token embeddings
        # combined_features = hidden_features + token_emb  # [B, L, d_model]
        # combined_features = self.layer_norm(combined_features)
        
        # # Add positional encoding
        # combined_features = combined_features.transpose(0, 1)  # [L, B, d_model]
        # combined_features = self.pos_encoding(combined_features)
        # combined_features = combined_features.transpose(0, 1)  # [B, L, d_model]
        
        # # Create causal mask for autoregressive prediction
        # causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # causal_mask = causal_mask.to(combined_features.device)
        
        # # Self-attention with causal mask
        # # Use the combined features as both memory and target for decoder
        # decoded_features = self.transformer_decoder(
        #     tgt=combined_features,
        #     memory=combined_features,
        #     tgt_mask=causal_mask,
        #     memory_mask=causal_mask
        # )  # [B, L, d_model]
        
        # # Predict CFG scales
        # cfg_scales_raw = self.cfg_scale_head(decoded_features).squeeze(-1)  # [B, L]
        
        # # Scale to desired range
        # cfg_scale_min, cfg_scale_max = self.cfg_scale_range
        # predicted_cfg_scales = cfg_scales_raw * (cfg_scale_max - cfg_scale_min) + cfg_scale_min
        
        # outputs = {
        #     'predicted_cfg_scales': predicted_cfg_scales,  # [B, L]
        # }
        
        return output["predicted_cfg_scales"], output["cache"]

    def forward_incremental(
        self,
        conditional_hidden: torch.Tensor,  # [batch_size, 1, hidden_dim] - current timestep only
        unconditional_hidden: torch.Tensor,  # [batch_size, 1, hidden_dim] - current timestep only
        previous_tokens: torch.Tensor,  # [batch_size, 1, hidden_dim] - current timestep only
        cache: Optional[Dict] = None,  # Cache for previous computations
        timestep: int = 0,  # Current timestep
    ) -> Dict[str, torch.Tensor]:
        """
        Incremental forward pass for training - only processes current timestep.
        Uses cached previous computations for efficiency.
        """
        
        batch_size = conditional_hidden.shape[0]
        
        # Project current hidden states to feature space
        # cond_features = self.conditional_proj(conditional_hidden)  # [B, 1, d_model//2]
        # uncond_features = self.unconditional_proj(unconditional_hidden)  # [B, 1, d_model//2]
        cond_features = conditional_hidden
        uncond_features = unconditional_hidden
        
        # Concatenate conditional and unconditional features
        hidden_features = torch.cat([cond_features, uncond_features], dim=-1)  # [B, 1, d_model]
        hidden_features = self.linear_proj(hidden_features)
        # Combine hidden state features and token embeddings
        combined_features = hidden_features + previous_tokens # [B, 1, d_model]
        combined_features = self.layer_norm(combined_features)
        
        # Add positional encoding for current timestep
        combined_features = combined_features.transpose(0, 1)  # [1, B, d_model]
        pos_encoding = self.pos_encoding.pe[timestep:timestep+1, :]  # [1, 1, d_model]
        combined_features = combined_features + pos_encoding.expand(-1, batch_size, -1)
        combined_features = combined_features.transpose(0, 1)  # [B, 1, d_model]
        
        # Initialize or update cache
        if cache is None:
            cache = {
                'past_key_values': [None] * len(self.transformer_decoder.layers),
                'past_features': []
            }
        
        # Store current features for future reference
        cache['past_features'].append(combined_features)
        
        # For true incremental decoding, we process layers one by one with KV caching
        current_input = combined_features  # [B, 1, d_model]
        
        # Process through each transformer layer with incremental KV caching
        for layer_idx, layer in enumerate(self.transformer_decoder.layers):
            # Get cached key-values for this layer
            past_kv = cache['past_key_values'][layer_idx]
            
            if past_kv is None:
                # First timestep - no cache yet
                # Create memory from current input for self-attention
                memory = current_input
                tgt_mask = None  # No mask needed for single token
                memory_mask = None  # No memory mask needed when memory is just the current token
            else:
                # Subsequent timesteps - use cached keys/values
                # Concatenate past and current for memory
                past_memory = past_kv['memory']  # [B, prev_len, d_model]
                memory = torch.cat([past_memory, current_input], dim=1)  # [B, prev_len+1, d_model]
                
                # Create causal mask only for the current position
                seq_len = memory.shape[1]
                # For a single target token, we don't need a tgt_mask (no self-attention between target tokens)
                tgt_mask = None
                # But we need a memory mask to control which source tokens we can attend to
                # Shape should match the implementation: either [nhead*batch_size, tgt_seq_len, src_seq_len]
                # or [batch_size, nhead, tgt_seq_len, src_seq_len] depending on the implementation
                nhead = self.transformer_decoder.layers[0].multihead_attn.num_heads
                # Create the base mask and expand it to match the expected shape
                # The PyTorch implementation likely expects batch_size*nhead as the first dimension
                memory_mask = torch.zeros(1, 1, seq_len).bool().to(current_input.device)
                memory_mask = memory_mask.expand(batch_size * nhead, -1, -1)  # [batch_size*nhead, 1, seq_len]
            
            # Apply the transformer layer
            # For incremental decoding, we only pass the current token as target
            layer_output = layer(
                tgt=current_input,      # [B, 1, d_model] - only current token
                memory=memory,          # [B, seq_len, d_model] - all tokens up to current
                tgt_mask=tgt_mask,      # No tgt_mask needed for single token input
                memory_mask=memory_mask # Control which source tokens we can attend to
            )  # [B, 1, d_model]
            
            # Update cache for this layer
            cache['past_key_values'][layer_idx] = {
                'memory': memory,  # Store the full memory for next timestep
            }
            
            # Output becomes input for next layer
            current_input = layer_output
        
        decoded_features = current_input  # [B, 1, d_model]
        
        # Predict CFG scale for current timestep
        cfg_scale_raw = self.cfg_scale_head(decoded_features).squeeze(-1)  # [B, 1]
        
        # Scale to desired range
        cfg_scale_min, cfg_scale_max = self.cfg_scale_range
        predicted_cfg_scale = cfg_scale_raw * (cfg_scale_max - cfg_scale_min) + cfg_scale_min
        
        outputs = {
            'predicted_cfg_scales': predicted_cfg_scale,  # [B, 1]
            'cache': cache,
        }
        
        return outputs

    def reset_cache(self):
        """Reset the cache when starting a new sequence."""
        return {
            'past_key_values': [None] * len(self.transformer_decoder.layers),
            'past_features': []
        }
    
    def predict_next_cfg_scale(
        self,
        conditional_hidden: torch.Tensor,  # [1, 1, hidden_dim] - current timestep
        unconditional_hidden: torch.Tensor,  # [1, 1, hidden_dim] - current timestep
        previous_tokens: torch.Tensor,  # [1, seq_len] - all previous tokens
        cache: Optional[Dict] = None,  # For efficient inference
    ) -> torch.Tensor:
        """
        Predict CFG scale for the next timestep during inference.
        This method is optimized for step-by-step token generation.
        """
        
        with torch.no_grad():
            # If this is the first token, initialize history
            if previous_tokens.shape[1] == 0:
                # Use a special start token or zero
                previous_tokens = torch.zeros(1, 1, dtype=torch.long, device=conditional_hidden.device)
            
            # For inference, we only need the last position
            seq_len = previous_tokens.shape[1]
            
            # Project current hidden states
            cond_features = self.conditional_proj(conditional_hidden)  # [1, 1, d_model//2]
            uncond_features = self.unconditional_proj(unconditional_hidden)  # [1, 1, d_model//2]
            hidden_features = torch.cat([cond_features, uncond_features], dim=-1)  # [1, 1, d_model]
            
            # Get token embedding for the last token
            token_emb = self.token_embedding(previous_tokens[:, -1:])  # [1, 1, d_model]
            
            # Combine features
            combined_features = hidden_features + token_emb
            combined_features = self.layer_norm(combined_features)
            
            # Add positional encoding (only for the current position)
            pos_idx = seq_len - 1
            combined_features = combined_features.transpose(0, 1)  # [1, 1, d_model]
            combined_features = combined_features + self.pos_encoding.pe[pos_idx:pos_idx+1, :].unsqueeze(1)
            combined_features = combined_features.transpose(0, 1)  # [1, 1, d_model]
            
            # For efficient inference, we could implement KV caching here
            # For now, use the simple approach
            decoded_features = self.transformer_decoder(
                tgt=combined_features,
                memory=combined_features,
            )  # [1, 1, d_model]
            
            # Predict CFG scale
            cfg_scale_raw = self.cfg_scale_head(decoded_features).squeeze(-1)  # [1, 1]
            
            # Scale to desired range
            cfg_scale_min, cfg_scale_max = self.cfg_scale_range
            predicted_cfg_scale = cfg_scale_raw * (cfg_scale_max - cfg_scale_min) + cfg_scale_min
            
            return predicted_cfg_scale.squeeze()  # Return scalar value


# External utility functions for applying CFG (used outside the model)

def apply_guided_sampling(
    conditional_logits: torch.Tensor,
    unconditional_logits: torch.Tensor,
    cfg_scale: float
) -> torch.Tensor:
    """
    Apply classifier-free guidance with the predicted CFG scale.
    Note: This function works with logits (after the output layer),
    as the final sampling is done based on token logits, not hidden states.
    
    This is an external utility function intended to be used after the CFG prediction.
    """
    # Convert to log probabilities
    logp_conditional = F.log_softmax(conditional_logits, dim=-1)
    logp_unconditional = F.log_softmax(unconditional_logits, dim=-1)
    
    # Apply CFG formula: logp_guided = logp_uncond + cfg_scale * (logp_cond - logp_uncond)
    guided_logp = logp_unconditional + cfg_scale * (logp_conditional - logp_unconditional)
    
    return guided_logp


def apply_guided_sampling_batch(
    conditional_logits: torch.Tensor,  # [B, L, V]
    unconditional_logits: torch.Tensor,  # [B, L, V]
    cfg_scales: torch.Tensor,  # [B, L]
) -> torch.Tensor:
    """
    Vectorized version of apply_guided_sampling for batch processing.
    Applies different CFG scales to each position in the batch.
    
    This is an external utility function intended to be used after the CFG prediction.
    
    Args:
        conditional_logits: Conditional logits with shape [batch_size, seq_len, vocab_size]
        unconditional_logits: Unconditional logits with shape [batch_size, seq_len, vocab_size]
        cfg_scales: CFG scales with shape [batch_size, seq_len]
        
    Returns:
        Guided logits with shape [batch_size, seq_len, vocab_size]
    """
    # Convert to log probabilities
    logp_conditional = F.log_softmax(conditional_logits, dim=-1)
    logp_unconditional = F.log_softmax(unconditional_logits, dim=-1)
    
    # Reshape cfg_scales for broadcasting
    cfg_scales = cfg_scales.unsqueeze(-1)  # [B, L, 1]
    
    # Apply CFG formula with broadcasting
    guided_logp = logp_unconditional + cfg_scales * (logp_conditional - logp_unconditional)
    
    return guided_logp
