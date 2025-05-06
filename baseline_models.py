from baseline_data_utils import MergedMelHarmDataset, PureGenCollator, SeparatedMelHarmDataset
import os
import numpy as np
from harmony_tokenizers_m21 import MelodyPitchTokenizer, ChordSymbolTokenizer, MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, BartForConditionalGeneration, BartConfig
import torch

melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
harmony_tokenizer = ChordSymbolTokenizer.from_pretrained('saved_tokenizers/ChordSymbolTokenizer')

baseline_tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)

def get_gpt2_and_dataset(
        model_path = 'baseline_models/saved_models/gpt/ChordSymbolTokenizer/ChordSymbolTokenizer.pt',
        device_name='cpu',
        data_dir = '/media/maindisk/maximos/data/hooktheory_all12_test'
    ):
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(baseline_tokenizer.vocab),
        n_positions=512,
        n_layer=8,
        n_head=16,
        pad_token_id=baseline_tokenizer.vocab[baseline_tokenizer.pad_token],
        bos_token_id=baseline_tokenizer.vocab[baseline_tokenizer.bos_token],
        eos_token_id=baseline_tokenizer.vocab[baseline_tokenizer.eos_token],
        n_embd=512
    )

    model_gpt = GPT2LMHeadModel(config)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)

    checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    model_gpt.load_state_dict(checkpoint)

    model_gpt.eval()
    model_gpt.to(device)

    gpt_dataset = MergedMelHarmDataset(data_dir, baseline_tokenizer, max_length=512, return_harmonization_labels=True, num_bars=16)

    return model_gpt, gpt_dataset
# end get_gpt2_and_dataset

def get_bart_and_dataset(
        model_path = 'baseline_models/saved_models/bart/ChordSymbolTokenizer/ChordSymbolTokenizer.pt',
        device_name='cpu',
        data_dir = '/media/maindisk/maximos/data/hooktheory_all12_test'
    ):
    bart_config = BartConfig(
        vocab_size=len(baseline_tokenizer.vocab),
        pad_token_id=baseline_tokenizer.pad_token_id,
        bos_token_id=baseline_tokenizer.bos_token_id,
        eos_token_id=baseline_tokenizer.eos_token_id,
        decoder_start_token_id=baseline_tokenizer.bos_token_id,
        forced_eos_token_id=baseline_tokenizer.eos_token_id,
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

    bart_model = BartForConditionalGeneration(bart_config)
    
    bart_dataset = SeparatedMelHarmDataset(data_dir, baseline_tokenizer, max_length=512, num_bars=16)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    
    checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    bart_model.load_state_dict(checkpoint)

    bart_model.eval()
    bart_model.to(device)
    return bart_model, bart_dataset
# end get_bart_and_dataset

def generate_save_with_gpt2_baseline(idx, file_name):
    gpt_model, gpt_dataset = get_gpt2_and_dataset()
    d = gpt_dataset[idx]
    start_harmony_position = np.where( d['input_ids'] == baseline_tokenizer.vocab['<h>'] )[0][0]
    input_ids = d['input_ids'][:(start_harmony_position+1)].to(gpt_model.device)

    outputs = gpt_model.generate(
        input_ids=input_ids.reshape(1, input_ids.shape[0]),
        eos_token_id=baseline_tokenizer.eos_token_id,
        max_length=gpt_model.config.max_position_embeddings,
        num_beams=5,
        do_sample=True,
        temperature=1
    )

    output_tokens = [baseline_tokenizer.ids_to_tokens[t] for t in outputs[0].tolist()]

    baseline_tokenizer.decode(output_tokens, output_format='file', output_path=file_name)
# end generate_save_with_gpt2_baseline

def generate_save_with_bart_baseline(idx, file_name):
    bart_model, bart_dataset = get_bart_and_dataset()
    d = bart_dataset[idx]
    input_ids = d['input_ids'].to(bart_model.device)

    outputs = bart_model.generate(
        input_ids=input_ids.reshape(1, input_ids.shape[0]),
        eos_token_id=baseline_tokenizer.eos_token_id,
        max_length=bart_model.config.max_position_embeddings,
        num_beams=5,
        do_sample=True,
        temperature=1
    )

    output_tokens = [baseline_tokenizer.ids_to_tokens[t] for t in input_ids.tolist()] + \
        [baseline_tokenizer.ids_to_tokens[t] for t in outputs[0].tolist()[1:]]
    baseline_tokenizer.decode(output_tokens, output_format='file', output_path=file_name)
# end generate_save_with_bart_baseline