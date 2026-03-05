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
import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.v100.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.v100.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for idx, i in enumerate(tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()


    def inference_instruct3(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True, cfg_scale=1.0, cfg_scale_prompt=1.0, cfg_scale_target=1.0, filter_topk=-1, drop_prompt=True, drop_target=False, cfg_rescale=1.0):  # Rescale should be smaller than cfg_scale
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for idx, i in enumerate(tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))):
            model_input_conditional, model_input_unconditional = self.frontend.frontend_instruct3(i, instruct_text, prompt_speech_16k, self.sample_rate)
            model_input_unconditional["cfg_scale"] = cfg_scale
            model_input_unconditional["dual_cfg_scale"] = [cfg_scale_prompt, cfg_scale_target]
            model_input_unconditional["filter_topk"] = filter_topk
            model_input_unconditional["drop_prompt"] = drop_prompt
            model_input_unconditional["drop_target"] = drop_target
            model_input_unconditional["cfg_rescale"] = cfg_rescale
            # if drop_prompt:
            #     assert drop_target is False, 'drop_prompt and drop_target cannot be True at the same time!'
            # elif drop_target:
            #     assert drop_prompt is False, 'drop_target and drop_prompt cannot be True at the same time!'
            if drop_prompt and drop_target:
                print("Perform dual-CFG with both prompt and target dropped.")
            assert not (drop_prompt == False and drop_target == False), 'Either drop_prompt or drop_target must be True for CFG!'
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input_conditional, stream=stream, speed=speed, unconditional_dict=model_input_unconditional):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
                
    def inference_instruct4(self, tts_text, instruct_text, opp_instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True, cfg_scale=1.0, filter_topk=-1, drop_prompt=True, drop_target=False, cfg_rescale=1.0):  # Rescale should be smaller than cfg_scale
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for idx, i in enumerate(tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))):
            model_input_conditional, model_input_unconditional = self.frontend.frontend_instruct4(i, instruct_text, opp_instruct_text, prompt_speech_16k, self.sample_rate)
            model_input_unconditional["cfg_scale"] = cfg_scale
            model_input_unconditional["filter_topk"] = filter_topk
            model_input_unconditional["drop_prompt"] = drop_prompt
            model_input_unconditional["drop_target"] = drop_target
            model_input_unconditional["cfg_rescale"] = cfg_rescale
            if drop_prompt:
                assert drop_target is False, 'drop_prompt and drop_target cannot be True at the same time!'
            elif drop_target:
                assert drop_prompt is False, 'drop_target and drop_prompt cannot be True at the same time!'
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input_conditional, stream=stream, speed=speed, unconditional_dict=model_input_unconditional):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
