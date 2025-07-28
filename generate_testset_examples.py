import torch
import torch.nn as nn
from models import GridMLMMelHarm, GridMLMMelHarmNoStage
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
from generate_utils import random_progressive_generate, structured_progressive_generate,\
    load_model, load_model_no_stage, overlay_generated_harmony, save_harmonized_score
import os
import numpy as np
from baseline_models import BaselineModeller

generate_primary = True
generate_baseline = True
generate_ablations = True

# output folders
mxl_folder = 'musicXMLs/testset/'
midi_folder = 'MIDIs/testset/'
os.makedirs('musicXMLs', exist_ok=True)
os.makedirs('MIDIs', exist_ok=True)
os.makedirs(mxl_folder, exist_ok=True)
os.makedirs(midi_folder, exist_ok=True)

# val_dir = '/media/maindisk/maximos/data/hooktheory_all12_test'
# val_dir = '/media/maindisk/maximos/data/hooktheory_test'
val_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_test'
tokenizer = CSGridMLMTokenizer(fixed_length=256)
tokenizer_noPCs = CSGridMLMTokenizer(fixed_length=256, use_pc_roll=False)
# val_dataset = CSGridMLMDataset(val_dir, tokenizer, 512)

if generate_baseline:
    baseline_model_base_path = 'baseline_models/saved_models/'
    bm = BaselineModeller(baseline_model_base_path, num_heads=8, data_dir = val_dir, device_name='cuda:2')

mask_token_id = tokenizer.mask_token_id
pad_token_id = tokenizer.pad_token_id
nc_token_id = tokenizer.nc_token_id

data_files = []
for dirpath, _, filenames in os.walk(val_dir):
    for file in filenames:
        if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') or \
            file.endswith('.mid') or file.endswith('.midi'):
            full_path = os.path.join(dirpath, file)
            data_files.append(full_path)
print('total data_files:', len(data_files))

# how many files to generate
num_files = len(data_files)

# get random indeces from 0 to len(data_files)-1
# random_indices = np.random.permutation(len(data_files))[:num_files]
random_indices = np.arange(num_files)
# random_indices = np.arange(511, 700, 1)
# random_indices = np.arange(1350, num_files, 1)
# random_indices = [2,97]
# random_indices = [1473]

# load models
if generate_primary:
    random10_model = load_model(curriculum_type='random', total_stages=10, subfolder='CA', device_name='cuda:2', tokenizer=tokenizer)
    random20_model = load_model(curriculum_type='random', total_stages=20, subfolder='CA', device_name='cuda:2', tokenizer=tokenizer)
    base2_model = load_model(curriculum_type='base2', subfolder='CA', device_name='cuda:2', tokenizer=tokenizer)
if generate_ablations:
    random10_no_stage_model = load_model_no_stage(curriculum_type='random', total_stages=10, subfolder='CA', device_name='cuda:2', tokenizer=tokenizer)
    random10_noPCs_model = load_model(curriculum_type='random', total_stages=10, subfolder='CA/noPCs', device_name='cuda:2', tokenizer=tokenizer_noPCs, pianoroll_dim=88)
    random20_no_stage_model = load_model_no_stage(curriculum_type='random', total_stages=20, subfolder='CA', device_name='cuda:2', tokenizer=tokenizer)
    random20_noPCs_model = load_model(curriculum_type='random', total_stages=20, subfolder='CA/noPCs', device_name='cuda:2', tokenizer=tokenizer_noPCs, pianoroll_dim=88)
    base2_no_stage_model = load_model_no_stage(curriculum_type='base2', subfolder='CA', device_name='cuda:2', tokenizer=tokenizer)
    base2_noPCs_model = load_model(curriculum_type='base2', subfolder='CA/noPCs', device_name='cuda:2', tokenizer=tokenizer_noPCs, pianoroll_dim=88)

for i,idx in enumerate(random_indices):
    print(f'{i+1}/{num_files} : {idx}{data_files[idx].replace(val_dir,'').replace('/','_')}')
    # base name to save generated file
    save_name_base = f'{data_files[idx].replace(val_dir,'').replace('/','_')}'
    # get encoded data from tokenizer
    encoded = tokenizer.encode(data_files[idx])
    encoded_noPCs = tokenizer_noPCs.encode(data_files[idx])
    melody_grid = torch.stack([torch.tensor(encoded['pianoroll'], dtype=torch.float)])
    melody_grid_noPCs = torch.stack([torch.tensor(encoded_noPCs['pianoroll'], dtype=torch.float)])
    conditioning_vec = torch.stack([torch.tensor(encoded['time_signature'], dtype=torch.float)])
    harmony_gt = torch.stack([torch.tensor(encoded['input_ids'], dtype=torch.float)])
    if generate_primary:
        # generate with random model
        random10_generated_harmony = random_progressive_generate(
            model=random10_model,
            melody_grid=melody_grid,
            conditioning_vec=conditioning_vec,
            num_stages=10,
            mask_token_id=tokenizer.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        random10_output_tokens = []
        for t in random10_generated_harmony[0].tolist():
            random10_output_tokens.append( tokenizer.ids_to_tokens[t] )
        # generate with random20 model
        random20_generated_harmony = random_progressive_generate(
            model=random20_model,
            melody_grid=melody_grid,
            conditioning_vec=conditioning_vec,
            num_stages=20,
            mask_token_id=tokenizer.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        random20_output_tokens = []
        for t in random20_generated_harmony[0].tolist():
            random20_output_tokens.append( tokenizer.ids_to_tokens[t] )
        # generate with base2 model
        base2_generated_harmony = structured_progressive_generate(
            model=base2_model,
            melody_grid=melody_grid,
            conditioning_vec=conditioning_vec,
            num_stages=10,
            mask_token_id=tokenizer.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        base2_output_tokens = []
        for t in base2_generated_harmony[0].tolist():
            base2_output_tokens.append( tokenizer.ids_to_tokens[t] )
        # keep ground truth
        harmony_gt_tokens = []
        for t in harmony_gt[0].tolist():
            harmony_gt_tokens.append( tokenizer.ids_to_tokens[t] )
        # make midi files
        # random10
        print(f'{i+1}/{num_files} : processing random10')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            random10_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_random10' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

        # random20
        print(f'{i+1}/{num_files} : processing random20')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            random20_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_random20' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)
        
        # base2
        print(f'{i+1}/{num_files} : processing base2')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            base2_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_base2' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)
        
        # real
        print(f'{i+1}/{num_files} : processing real')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            harmony_gt_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_real' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

    if generate_baseline:
        midi_file_name = midi_folder + f'{idx}_gpt2' + save_name_base + '.mid'
        print(f'baseline gpt2 {i+1}/{num_files} : saving midi')
        bm.generate_save_with_gpt2_baseline(idx, midi_file_name, input_melody_part=encoded['melody_part'])

        midi_file_name = midi_folder + f'{idx}_bart' + save_name_base + '.mid'
        print(f'baseline bart {i+1}/{num_files} : saving midi')
        bm.generate_save_with_bart_baseline(idx, midi_file_name, input_melody_part=encoded['melody_part'])
    if generate_ablations:
        # generate with random10 no stage model
        random10_no_stage_generated_harmony = random_progressive_generate(
            model=random10_no_stage_model,
            melody_grid=melody_grid,
            conditioning_vec=conditioning_vec,
            num_stages=10,
            mask_token_id=tokenizer.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        random10_no_stage_output_tokens = []
        for t in random10_no_stage_generated_harmony[0].tolist():
            random10_no_stage_output_tokens.append( tokenizer.ids_to_tokens[t] )
        
        print(f'{i+1}/{num_files} : processing random10 no stage')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            random10_no_stage_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_random10NS' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

        # generate with random20 no stage model
        random20_no_stage_generated_harmony = random_progressive_generate(
            model=random20_no_stage_model,
            melody_grid=melody_grid,
            conditioning_vec=conditioning_vec,
            num_stages=20,
            mask_token_id=tokenizer.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        random20_no_stage_output_tokens = []
        for t in random20_no_stage_generated_harmony[0].tolist():
            random20_no_stage_output_tokens.append( tokenizer.ids_to_tokens[t] )
        
        print(f'{i+1}/{num_files} : processing random20 no stage')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            random20_no_stage_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_random20NS' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

        # generate with base2 no stage model
        base2_no_stage_generated_harmony = structured_progressive_generate(
            model=base2_no_stage_model,
            melody_grid=melody_grid,
            conditioning_vec=conditioning_vec,
            num_stages=10,
            mask_token_id=tokenizer.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        base2_no_stage_output_tokens = []
        for t in base2_no_stage_generated_harmony[0].tolist():
            base2_no_stage_output_tokens.append( tokenizer.ids_to_tokens[t] )
        
        print(f'{i+1}/{num_files} : processing base2 no stage')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            base2_no_stage_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_base2NS' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

        # generate with random10 no PCs model
        random10_noPCs_generated_harmony = random_progressive_generate(
            model=random10_noPCs_model,
            melody_grid=melody_grid_noPCs,
            conditioning_vec=conditioning_vec,
            num_stages=10,
            mask_token_id=tokenizer_noPCs.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        random10_noPCs_output_tokens = []
        for t in random10_noPCs_generated_harmony[0].tolist():
            random10_noPCs_output_tokens.append( tokenizer_noPCs.ids_to_tokens[t] )
        
        print(f'{i+1}/{num_files} : processing random10 noPCs')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            random10_noPCs_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_random10NPCs' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

        # generate with random20 no PCs model
        random20_noPCs_generated_harmony = random_progressive_generate(
            model=random20_noPCs_model,
            melody_grid=melody_grid_noPCs,
            conditioning_vec=conditioning_vec,
            num_stages=20,
            mask_token_id=tokenizer_noPCs.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        random20_noPCs_output_tokens = []
        for t in random20_noPCs_generated_harmony[0].tolist():
            random20_noPCs_output_tokens.append( tokenizer_noPCs.ids_to_tokens[t] )
        
        print(f'{i+1}/{num_files} : processing random20 noPCs')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            random20_noPCs_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_random20NPCs' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)

        # generate with base2 no PCs model
        base2_noPCs_generated_harmony = structured_progressive_generate(
            model=base2_noPCs_model,
            melody_grid=melody_grid_noPCs,
            conditioning_vec=conditioning_vec,
            num_stages=10,
            mask_token_id=tokenizer_noPCs.mask_token_id,
            temperature=1.0,
            strategy='sample',
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True         # disallow <pad>/<nc> before melody ends
        )
        base2_noPCs_output_tokens = []
        for t in base2_noPCs_generated_harmony[0].tolist():
            base2_noPCs_output_tokens.append( tokenizer_noPCs.ids_to_tokens[t] )
        
        print(f'{i+1}/{num_files} : processing base2 noPCs')
        score = overlay_generated_harmony(
            encoded['melody_part'],
            base2_noPCs_output_tokens,
            encoded['ql_per_quantum'],
            encoded['skip_steps']
        )
        midi_file_name = midi_folder + f'{idx}_base2NPCs' + save_name_base + '.mid'
        save_harmonized_score(score, out_path=midi_file_name)