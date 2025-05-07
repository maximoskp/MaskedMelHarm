from baseline_data_utils import MergedMelHarmDataset, PureGenCollator, SeparatedMelHarmDataset
import os
import numpy as np
from harmony_tokenizers_m21 import MelodyPitchTokenizer, ChordSymbolTokenizer, MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, BartForConditionalGeneration, BartConfig
import torch


class BaselineModeller():
    def __init__(self, data_dir = '/media/maindisk/maximos/data/hooktheory_all12_test', device_name='cpu'):
        melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
        harmony_tokenizer = ChordSymbolTokenizer.from_pretrained('saved_tokenizers/ChordSymbolTokenizer')
        self.baseline_tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)
        self.device_name = device_name
        self.data_dir = data_dir
        self.make_gpt2_and_dataset()
        self.make_bart_and_dataset()

    def make_gpt2_and_dataset(
        self,
        model_path = 'baseline_models/saved_models/gpt/ChordSymbolTokenizer/ChordSymbolTokenizer.pt',
    ):
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(self.baseline_tokenizer.vocab),
            n_positions=512,
            n_layer=8,
            n_head=16,
            pad_token_id=self.baseline_tokenizer.vocab[self.baseline_tokenizer.pad_token],
            bos_token_id=self.baseline_tokenizer.vocab[self.baseline_tokenizer.bos_token],
            eos_token_id=self.baseline_tokenizer.vocab[self.baseline_tokenizer.eos_token],
            n_embd=512
        )

        self.gpt_model = GPT2LMHeadModel(config)

        if self.device_name == 'cpu':
            device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device = torch.device(self.device_name)
            else:
                print('Selected device not available: ' + self.device_name)

        checkpoint = torch.load(model_path, map_location=self.device_name, weights_only=True)
        self.gpt_model.load_state_dict(checkpoint)

        self.gpt_model.eval()
        self.gpt_model.to(device)

        self.gpt_dataset = MergedMelHarmDataset(
            self.data_dir, 
            self.baseline_tokenizer, 
            max_length=512, 
            return_harmonization_labels=True, num_bars=16
        )
    # end make_gpt2_and_dataset

    def make_bart_and_dataset(
        self,
        model_path = 'baseline_models/saved_models/bart/ChordSymbolTokenizer/ChordSymbolTokenizer.pt',
    ):
        bart_config = BartConfig(
            vocab_size=len(self.baseline_tokenizer.vocab),
            pad_token_id=self.baseline_tokenizer.pad_token_id,
            bos_token_id=self.baseline_tokenizer.bos_token_id,
            eos_token_id=self.baseline_tokenizer.eos_token_id,
            decoder_start_token_id=self.baseline_tokenizer.bos_token_id,
            forced_eos_token_id=self.baseline_tokenizer.eos_token_id,
            max_position_embeddings=512,
            encoder_layers=8,
            encoder_attention_heads=16,
            encoder_ffn_dim=512,
            decoder_layers=8,
            decoder_attention_heads=16,
            decoder_ffn_dim=512,
            d_model=512,
            encoder_layerdrop=0.25,
            decoder_layerdrop=0.25,
            dropout=0.25
        )

        self.bart_model = BartForConditionalGeneration(bart_config)

        self.bart_dataset = SeparatedMelHarmDataset(
            self.data_dir, 
            self.baseline_tokenizer, 
            max_length=512, 
            num_bars=16)

        if self.device_name == 'cpu':
            device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device = torch.device(self.device_name)
            else:
                print('Selected device not available: ' + self.device_name)

        checkpoint = torch.load(model_path, map_location=self.device_name, weights_only=True)
        self.bart_model.load_state_dict(checkpoint)

        self.bart_model.eval()
        self.bart_model.to(device)
    # end make_bart_and_dataset

    def generate_save_with_gpt2_baseline(self, idx, file_name):
        d = self.gpt_dataset[idx]
        start_harmony_position = np.where( d['input_ids'] == self.baseline_tokenizer.vocab['<h>'] )[0][0]
        input_ids = d['input_ids'][:(start_harmony_position+1)].to(self.gpt_model.device)

        outputs = self.gpt_model.generate(
            input_ids=input_ids.reshape(1, input_ids.shape[0]),
            eos_token_id=self.baseline_tokenizer.eos_token_id,
            max_length=self.gpt_model.config.max_position_embeddings,
            num_beams=5,
            do_sample=True,
            temperature=1
        )

        output_tokens = [self.baseline_tokenizer.ids_to_tokens[t] for t in outputs[0].tolist()]

        self.baseline_tokenizer.decode(output_tokens, output_format='file', output_path=file_name)
    # end generate_save_with_gpt2_baseline

    def generate_save_with_bart_baseline(self, idx, file_name):
        d = self.bart_dataset[idx]
        input_ids = d['input_ids'].to(self.bart_model.device)

        outputs = self.bart_model.generate(
            input_ids=input_ids.reshape(1, input_ids.shape[0]),
            eos_token_id=self.baseline_tokenizer.eos_token_id,
            max_length=self.bart_model.config.max_position_embeddings,
            num_beams=5,
            do_sample=True,
            temperature=1
        )

        output_tokens = [self.baseline_tokenizer.ids_to_tokens[t] for t in input_ids.tolist()] + \
            [self.baseline_tokenizer.ids_to_tokens[t] for t in outputs[0].tolist()[1:]]
        self.baseline_tokenizer.decode(output_tokens, output_format='file', output_path=file_name)
    # end generate_save_with_bart_baseline
# end class BaselineModeller