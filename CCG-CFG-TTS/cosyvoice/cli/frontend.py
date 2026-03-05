# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
import time
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options = [
            {
                'device_id': 0,
            }
        ]
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"], provider_options=provider_options)
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len
    
    def _extract_text_token_batch(self, texts):
        # Most tokenizers (like transformers tokenizers) support native batch encoding
        # This is significantly faster than processing texts one by one
        
        # Check if the tokenizer has a batch encoding method
        if hasattr(self.tokenizer, 'batch_encode_plus'):
            # Use the tokenizer's native batch functionality (transformers-style)
            encoding = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                return_tensors='pt',
                allowed_special=self.allowed_special
            )
            text_token_tensor = encoding['input_ids'].to(dtype=torch.int32, device=self.device)
            text_token_lens = encoding['attention_mask'].sum(dim=1).to(dtype=torch.int32, device=self.device)
            return text_token_tensor, text_token_lens
            
        # For tokenizers that support encode_batch (some custom tokenizers)
        elif hasattr(self.tokenizer, 'encode_batch'):
            # Use the tokenizer's native batch functionality
            tokens_batch = self.tokenizer.encode_batch(texts, allowed_special=self.allowed_special)
            
            # Extract tokens and lengths
            text_tokens = [t.ids if hasattr(t, 'ids') else t for t in tokens_batch]
            text_token_lens = [len(t) for t in text_tokens]
            
            # Find the maximum length for padding
            max_len = max(text_token_lens)
            
            # Pad all tokens to the same length
            padded_tokens = []
            for token in text_tokens:
                padding = [0] * (max_len - len(token))
                padded_tokens.append(token + padding)
            
            # Convert to tensors
            text_token_tensor = torch.tensor(padded_tokens, dtype=torch.int32).to(self.device)
            text_token_len_tensor = torch.tensor(text_token_lens, dtype=torch.int32).to(self.device)
            
            return text_token_tensor, text_token_len_tensor
            
        else:
            # Fallback to sequential processing for tokenizers without batch support
            text_tokens = []
            text_token_lens = []
            
            for text in texts:
                token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
                text_tokens.append(token)
                text_token_lens.append(len(token))
            
            # Find the maximum length for padding
            max_len = max(text_token_lens)
            
            # Pad all tokens to the same length
            padded_tokens = []
            for token in text_tokens:
                padding = [0] * (max_len - len(token))
                padded_tokens.append(token + padding)
            
            # Convert to tensors
            text_token_tensor = torch.tensor(padded_tokens, dtype=torch.int32).to(self.device)
            text_token_len_tensor = torch.tensor(text_token_lens, dtype=torch.int32).to(self.device)
            
            return text_token_tensor, text_token_len_tensor

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len
    
    def _extract_speech_token_batch(self, speeches):
        """Batch version of _extract_speech_token to process multiple speech inputs at once"""
        speech_tokens = []
        speech_token_lens = []
        
        for speech in speeches:
            assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
            feat = whisper.log_mel_spectrogram(speech, n_mels=128)
            token = self.speech_tokenizer_session.run(None,
                                                     {self.speech_tokenizer_session.get_inputs()[0].name:
                                                      feat.detach().cpu().numpy(),
                                                      self.speech_tokenizer_session.get_inputs()[1].name:
                                                      np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
            speech_tokens.append(token)
            speech_token_lens.append(len(token))
        
        # Find the maximum length for padding
        max_len = max(speech_token_lens)
        
        # Pad all tokens to the same length
        padded_tokens = []
        for token in speech_tokens:
            padding = [0] * (max_len - len(token))
            padded_tokens.append(token + padding)
        
        # Convert to tensors
        speech_token_tensor = torch.tensor(padded_tokens, dtype=torch.int32).to(self.device)
        speech_token_len_tensor = torch.tensor(speech_token_lens, dtype=torch.int32).to(self.device)
        
        return speech_token_tensor, speech_token_len_tensor

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding
    
    def _extract_spk_embedding_batch(self, speeches):
        """Batch version of _extract_spk_embedding to process multiple speech inputs at once"""
        embeddings = []
        
        for speech in speeches:
            feat = kaldi.fbank(speech,
                              num_mel_bins=80,
                              dither=0,
                              sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            embedding = self.campplus_session.run(None,
                                                 {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
            embeddings.append(embedding)
        
        # Convert to tensor
        embeddings_tensor = torch.tensor(embeddings).to(self.device)
        return embeddings_tensor

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len
    
    def _extract_speech_feat_batch(self, speeches):
        """Batch version of _extract_speech_feat to process multiple speech inputs at once"""
        speech_feats = []
        speech_feat_lens = []
        
        for speech in speeches:
            # Extract features for this speech sample
            feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
            feat = feat.unsqueeze(dim=0)
            feat_len = feat.shape[1]
            
            speech_feats.append(feat)
            speech_feat_lens.append(feat_len)
        
        # Find max length for padding if needed
        max_len = max(speech_feat_lens)
        
        # If the features have different lengths, we need to pad them
        padded_feats = []
        for feat, feat_len in zip(speech_feats, speech_feat_lens):
            if feat_len < max_len:
                # Calculate padding needed
                padding = torch.zeros(1, max_len - feat_len, feat.shape[2], device=self.device)
                padded_feat = torch.cat([feat, padding], dim=1)
                padded_feats.append(padded_feat)
            else:
                padded_feats.append(feat)
        
        # Concatenate all features along the batch dimension
        batched_feats = torch.cat(padded_feats, dim=0)
        batched_feat_lens = torch.tensor(speech_feat_lens, dtype=torch.int32).to(self.device)
        
        return batched_feats, batched_feat_lens

    def text_normalize(self, text, split=True, text_frontend=True):
        if text_frontend is False:
            return [text] if split is True else text
        text = text.strip()
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input
    
    def frontend_zero_shot_batchfy(self, tts_texts, prompt_texts):
        """
        Batched version of frontend_zero_shot to process multiple inputs at once.
        Uses batched helper methods for improved efficiency.
        
        Args:
            tts_texts: List of text strings to synthesize
            prompt_texts: List of prompt text strings
            prompt_speeches_16k: List of prompt speech tensors at 16kHz
            resample_rate: Target resample rate
            
        Returns:
            Dictionary with batched model inputs
        """
        batch_size = len(tts_texts)
        assert batch_size == len(prompt_texts) and batch_size == len(tts_texts), \
            "All input lists must have the same length"
        
        # Extract text tokens for tts_texts and prompt_texts using batch methods
        tts_text_tokens, tts_text_token_lens = self._extract_text_token_batch(tts_texts)
        prompt_text_tokens, prompt_text_token_lens = self._extract_text_token_batch(prompt_texts)
        
        # Resample all prompt speeches to target rate
        # prompt_speeches_resampled = []
        # for speech in prompt_speeches_16k:
        #     resampled = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(speech)
        #     prompt_speeches_resampled.append(resampled)
        
        # # Extract speech features and tokens using batch methods
        # speech_feats, speech_feat_lens = self._extract_speech_feat_batch(prompt_speeches_resampled)
        # speech_tokens, speech_token_lens = self._extract_speech_token_batch(prompt_speeches_16k)
        
        # Handle 24000 Hz resampling case
        # if resample_rate == 24000:
        #     # We need to process each sample individually for the token_len adjustment
        #     for i in range(batch_size):
        #         # Calculate token_len for each sample
        #         token_len = min(int(speech_feats[i].shape[1] / 2), speech_tokens[i].shape[0])
                
        #         # Truncate features and tokens
        #         if i == 0:
        #             # Initialize new tensors for truncated data
        #             new_speech_feats = speech_feats[i:i+1, :2 * token_len].clone()
        #             new_speech_tokens = speech_tokens[i:i+1, :token_len].clone()
        #             new_speech_feat_lens = torch.tensor([2 * token_len], dtype=torch.int32, device=self.device)
        #             new_speech_token_lens = torch.tensor([token_len], dtype=torch.int32, device=self.device)
        #         else:
        #             # Append to existing tensors
        #             new_speech_feats = torch.cat([new_speech_feats, speech_feats[i:i+1, :2 * token_len]])
        #             new_speech_tokens = torch.cat([new_speech_tokens, speech_tokens[i:i+1, :token_len]])
        #             new_speech_feat_lens = torch.cat([new_speech_feat_lens, torch.tensor([2 * token_len], dtype=torch.int32, device=self.device)])
        #             new_speech_token_lens = torch.cat([new_speech_token_lens, torch.tensor([token_len], dtype=torch.int32, device=self.device)])
            
        #     # Replace with truncated tensors
        #     speech_feats, speech_feat_lens = new_speech_feats, new_speech_feat_lens
        #     speech_tokens, speech_token_lens = new_speech_tokens, new_speech_token_lens
        
        # Extract speaker embeddings using batch method
        # embeddings = self._extract_spk_embedding_batch(prompt_speeches_16k)
        
        # Create the model input dictionary
        model_input = {
            'text': tts_text_tokens, 
            'text_len': tts_text_token_lens,
            'prompt_text': prompt_text_tokens, 
            'prompt_text_len': prompt_text_token_lens,
            'llm_prompt_speech_token': None, 
            'llm_prompt_speech_token_len': None,
            'flow_prompt_speech_token': None, 
            'flow_prompt_speech_token_len': None,
            'prompt_speech_feat': None, 
            'prompt_speech_feat_len': None,
            'llm_embedding': None, 
            'flow_embedding': None
        }
        
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input
    
    def frontend_instruct3(self, tts_text, instruct_text, prompt_speech_16k, resample_rate):
        model_input_conditional = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate)
        model_input_unconditional = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate) # Or? self.frontend_zero_shot(tts_text, 'Transcription?', prompt_speech_16k, resample_rate)
        del model_input_conditional['llm_prompt_speech_token']
        del model_input_conditional['llm_prompt_speech_token_len']
        del model_input_unconditional['llm_prompt_speech_token']
        del model_input_unconditional['llm_prompt_speech_token_len']
        return model_input_conditional, model_input_unconditional
    
    def frontend_instruct4(self, tts_text, instruct_text, opp_instruction_text, prompt_speech_16k, resample_rate):
        model_input_conditional = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate)
        model_input_unconditional = self.frontend_zero_shot(tts_text, opp_instruction_text + '<|endofprompt|>', prompt_speech_16k, resample_rate) # Or? self.frontend_zero_shot(tts_text, 'Transcription?', prompt_speech_16k, resample_rate)
        del model_input_conditional['llm_prompt_speech_token']
        del model_input_conditional['llm_prompt_speech_token_len']
        del model_input_unconditional['llm_prompt_speech_token']
        del model_input_unconditional['llm_prompt_speech_token_len']
        return model_input_conditional, model_input_unconditional
    
    def frontend_instruct3_batchfy(self, tts_texts, instruct_texts):
        """
        Batched version of frontend_instruct3 to process multiple inputs at once.
        This implementation uses the batched frontend_zero_shot_batchfy for better efficiency.
        
        Args:
            tts_texts: List of text strings to synthesize
            instruct_texts: List of instruction texts
            prompt_speeches_16k: List of prompt speech tensors at 16kHz
            resample_rate: Target resample rate
            
        Returns:
            Tuple of conditional and unconditional model inputs with batched tensors
        """
        # Create a list of instruction texts with <|endofprompt|> appended
        instruct_texts_with_eop = [text + '<|endofprompt|>' for text in instruct_texts]
        
        # Create a list of empty strings for unconditional generation
        # empty_instruct_texts = [''] * len(tts_texts)
        
        # Process conditional and unconditional inputs in batch
        model_input_conditional = self.frontend_zero_shot_batchfy(
            tts_texts, 
            instruct_texts_with_eop, 
        )
        
        # model_input_unconditional = self.frontend_zero_shot_batchfy(
        #     tts_texts,
        #     empty_instruct_texts,
        #     prompt_speeches_16k,
        #     resample_rate
        # )
        
        # Remove llm_prompt_speech_token and llm_prompt_speech_token_len fields
        del model_input_conditional['llm_prompt_speech_token']
        del model_input_conditional['llm_prompt_speech_token_len']
        # del model_input_unconditional['llm_prompt_speech_token']
        # del model_input_unconditional['llm_prompt_speech_token_len']
        
        return model_input_conditional # , model_input_unconditional

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input
