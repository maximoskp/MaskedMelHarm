import torch
import torch.nn.functional as F
from train_utils import apply_structured_masking
from music21 import harmony, stream, metadata, chord, note, key, meter, tempo, duration
import mir_eval
import numpy as np
from copy import deepcopy
from models import GridMLMMelHarm, GridMLMMelHarmNoStage
import os
from music_utils import transpose_score

def remove_conflicting_rests(flat_part):
    """
    Remove any Rest in a flattened part whose offset coincides with a Note.
    Assumes the input stream is already flattened.
    This also tries to fix broken duration values that have come as a result of
    flattening.
    """
    cleaned = stream.Part()
    all_notes = [el for el in flat_part if isinstance(el, note.Note)]
    note_offsets = [n.offset for n in all_notes]
    note_durations = [n.offset for n in all_notes]
    for i, n in enumerate(all_notes):
        if n.duration.quarterLength == 0:
            if i < len(all_notes)-1:
                n.duration = duration.Duration( note_offsets[i+1] - note_offsets[i] )
            else:
                n.duration = duration.Duration( 0.5 )
        # if i < len(all_notes)-1:
        #     if note_durations[i] > note_offsets[i+1] - note_offsets[i]:
        #         n.duration = duration.Duration( note_offsets[i+1] - note_offsets[i] )

    for el in flat_part:
        # Skip Rest if it shares offset with a Note
        if isinstance(el, note.Rest) and el.offset in note_offsets:
            continue
        cleaned.insert(el.offset, el)

    return cleaned
# end remove_conflicting_rests

def random_progressive_generate(
    model,
    melody_grid,            # (1, seq_len, input_dim)
    conditioning_vec,       # (1, cond_dim)
    num_stages,             # e.g., 10
    mask_token_id,          # token ID used for masking
    temperature=1.0,        # optional temperature for sampling
    strategy='topk',        # 'topk' or 'sample' strategy for selecting new tokens
    pad_token_id=None,      # token ID for <pad>
    nc_token_id=None,       # token ID for <nc>
    force_fill=True,        # disallow <pad>/<nc> before melody ends
    chord_constraints=None  # take input harmony as constraints or generate from scratch
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]
    tokens_per_stage = int(seq_len / num_stages + 0.5)

    # Start with all tokens masked
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if chord_constraints is not None:
        idxs  = torch.logical_and( chord_constraints != nc_token_id , chord_constraints != pad_token_id )
        visible_harmony[ idxs ] = chord_constraints[idxs]
    
    # Find the last index in melody_grid that contains a non-zero value
    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)  # shape: (seq_len,)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except:
            last_active_index = -1
    else:
        last_active_index = -1  # Don't clamp anything if not forced
    for stage in range(num_stages):
        # Check for early stopping
        if not (visible_harmony == mask_token_id).any():
            break  # All tokens revealed
        with torch.no_grad():
            logits = model(
                melody_grid=melody_grid.to(model.device),
                conditioning_vec=conditioning_vec.to(model.device),
                harmony_tokens=visible_harmony.to(model.device),
                stage_indices=torch.LongTensor([stage]).to(model.device)
            )  # (1, seq_len, vocab_size)
        
        if force_fill and (pad_token_id is not None and nc_token_id is not None):
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float('-inf')
                    logits[0, i, nc_token_id] = float('-inf')
                else:
                    logits[0, i, :] = float('-inf')
                    logits[0, i, pad_token_id] = 1.0

        probs = torch.softmax(logits / temperature, dim=-1)  # (1, seq_len, vocab_size)
        confidences, predictions = torch.max(probs, dim=-1)  # (1, seq_len)

        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break  # All tokens revealed

        topk = min(tokens_per_stage, masked_positions.size(0))

        if strategy == 'topk':
            masked_confidences = confidences[0, masked_positions]
            topk_indices = torch.topk(masked_confidences, k=topk).indices
            selected_positions = masked_positions[topk_indices.to(device)]
        elif strategy == 'sample':
            # Sample `topk` indices from masked positions with probability proportional to confidence
            masked_confidences = confidences[0, masked_positions]
            p = masked_confidences / masked_confidences.sum()
            selected_positions = masked_positions[torch.multinomial(p, topk, replacement=False).to(device)]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # Update visible_harmony with selected predictions
        for idx in selected_positions:
            visible_harmony[0, idx] = predictions[0, idx]

    return visible_harmony  # Final generated token sequence
# end random_progressive_generate

def structured_progressive_generate(
    model,
    melody_grid,            # (1, seq_len, input_dim)
    conditioning_vec,       # (1, cond_dim)
    num_stages,             # e.g., 9 for 256 tokens
    mask_token_id,          # token ID used for masking
    temperature=1.0,        # optional temperature for sampling
    strategy='topk',        # 'topk' or 'sample'
    pad_token_id=None,      # token ID for <pad>
    nc_token_id=None,       # token ID for <nc>
    force_fill=True,        # disallow <pad>/<nc> before melody ends,
    chord_constraints=None  # take input harmony as constraints or generate from scratch
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if chord_constraints is not None:
        idxs  = torch.logical_and( chord_constraints != nc_token_id , chord_constraints != pad_token_id )
        visible_harmony[ idxs ] = chord_constraints[idxs]
    
    # Find the last index in melody_grid that contains a non-zero value
    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)  # shape: (seq_len,)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except:
            last_active_index = -1
    else:
        last_active_index = -1  # Don't clamp anything if not forced
    for stage in range(num_stages):
        # Check for early stopping
        if not (visible_harmony == mask_token_id).any():
            break  # All tokens revealed

        spacing_target = max(1, 2 ** (num_stages - stage - 1))  # e.g., 256 → 128 → ... → 1
        candidate_positions = torch.arange(0, seq_len, spacing_target, device=device)
        masked_positions = (visible_harmony[0] == mask_token_id).nonzero(as_tuple=True)[0]
        positions_to_predict = [pos for pos in candidate_positions if pos in masked_positions]

        if not positions_to_predict:
            continue  # Nothing to predict at this stage

        with torch.no_grad():
            logits = model(
                melody_grid=melody_grid.to(model.device),
                conditioning_vec=conditioning_vec.to(model.device),
                harmony_tokens=visible_harmony.to(model.device),
                stage_indices=torch.LongTensor([stage]).to(model.device)
            )  # (1, seq_len, vocab_size)

        if force_fill and (pad_token_id is not None and nc_token_id is not None):
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float('-inf')
                    logits[0, i, nc_token_id] = float('-inf')
                else:
                    logits[0, i, :] = float('-inf')
                    logits[0, i, pad_token_id] = 1.0
        
        probs = torch.softmax(logits / temperature, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)  # (1, seq_len)

        if strategy == 'topk':
            for pos in positions_to_predict:
                visible_harmony[0, pos] = predictions[0, pos]
        elif strategy == 'sample':
            for pos in positions_to_predict:
                prob_dist = probs[0, pos]
                sampled_token = torch.multinomial(prob_dist, num_samples=1)
                visible_harmony[0, pos] = sampled_token
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    return visible_harmony
# end structured_progressive_generate

def overlay_generated_harmony(melody_part, generated_chords, ql_per_16th, skip_steps):
    # create a part for chords in midi format
    # melody_part = melody_part.makeMeasures()
    # chords_part = deepcopy(melody_part)
    # Create deep copy of flat melody part
    # Create a new part for filtered content
    filtered_part = stream.Part()
    filtered_part.id = melody_part.id  # Preserve ID

    # Copy key and time signatures from the original part
    for el in melody_part.recurse().getElementsByClass((key.KeySignature, meter.TimeSignature,  tempo.MetronomeMark)):
        if el.offset < 64:
            filtered_part.insert(el.offset, el)

    # Copy notes and rests with offset < 64
    for el in melody_part.flatten().notesAndRests:
        if el.offset < 64:
            filtered_part.insert(el.offset, el)
    # clear conflicting rests with notes of the same offset
    filtered_part = remove_conflicting_rests(filtered_part)

    # Replace the original part with the filtered one
    melody_part = filtered_part
    
    # Remove old chord symbols
    for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
        melody_part.remove(el)
    
    # Prepare for clamping durations — convert melody to measures
    melody_measures = melody_part.makeMeasures()

    chords_part = deepcopy(melody_measures)
    # Strip musical elements but retain structure
    for measure in chords_part.getElementsByClass(stream.Measure):
        for el in list(measure):
            if isinstance(el, (note.Note, note.Rest, chord.Chord, harmony.ChordSymbol)):
                measure.remove(el)
        # Add a placeholder full-measure rest to preserve the measure
        full_rest = note.Rest()
        full_rest.quarterLength = measure.barDuration.quarterLength
        measure.insert(0.0, full_rest)

    # Track inserted chords
    last_chord_symbol = None
    inserted_chords = {}

    for i, mir_chord in enumerate(generated_chords):
        if mir_chord in ("<pad>", "<nc>"):
            continue
        if mir_chord == last_chord_symbol:
            continue
        
        offset = (i + skip_steps) * ql_per_16th

        # Decode mir_eval chord symbol to chord symbol object
        try:
            r, t, _ = mir_eval.chord.encode(mir_chord, reduce_extended_chords=True)
            pcs = r + np.where(t > 0)[0] + 48
            c = chord.Chord(pcs.tolist())
            chord_symbol = harmony.chordSymbolFromChord(c)
        except Exception as e:
            print(f"Skipping invalid chord {mir_chord} at step {i}: {e}")
            continue
        
        # Clamp duration so it doesn't overflow into next bar
        bars = list(chords_part.getElementsByClass(stream.Measure))
        for b in reversed(bars):
            if b.offset <= offset:
                bar = b
                break
        # bar = next((b for b in reversed(bars) if b.offset <= offset), None)

        offset = (i + skip_steps) * ql_per_16th
        
        if bar:
            bar_start = bar.offset
            bar_end = bar_start + bar.barDuration.quarterLength
            max_duration = bar_end - offset
            c.quarterLength = min(c.quarterLength, max_duration)
            # chord_symbol.quarterLength = min(c.quarterLength, max_duration)
        # chords_part.insert(offset, c)
        # Remove any placeholder rests at 0.0
        for el in bar.getElementsByOffset(0.0):
            if isinstance(el, note.Rest):
                bar.remove(el)
        bar.insert(offset - bar_start, c)
        # harmonized_part.insert(offset, chord_symbol)
        inserted_chords[i] = chord_symbol
        last_chord_symbol = mir_chord

    # Repeat previous chord at start of bars with no chord
    for m in chords_part.getElementsByClass(stream.Measure):
        bar_offset = m.offset
        bar_duration = m.barDuration.quarterLength
        # has_chord = any(isinstance(el, chord.Chord) and el.offset == bar_offset for el in m)
        # has_chord = any( isinstance(el, chord.Chord) for el in m )
        has_chord = any(isinstance(el, chord.Chord) and el.offset == 0. for el in m)
        if not has_chord:
            # Find previous chord before this measure
            prev_chords = []
            for curr_bar in chords_part.recurse().getElementsByClass(stream.Measure):
                for el in curr_bar.recurse().getElementsByClass(chord.Chord):
                        if curr_bar.offset + el.offset < bar_offset:
                            prev_chords.append(el)
            if prev_chords:
                # Remove any placeholder rests at 0.0
                for el in m.getElementsByOffset(0.0):
                    if isinstance(el, note.Rest):
                        m.remove(el)
                prev_chord = prev_chords[-1]
                m.insert(0.0, deepcopy(prev_chord))
        else:
            # Remove any placeholder rests at 0.0
            for el in m.getElementsByOffset(0.0):
                if isinstance(el, note.Rest):
                    m.remove(el)
            # modify duration so that it doesn't affect the next bar
            for el in m.notes:
                if isinstance(el, chord.Chord):
                    max_duration = bar_duration - el.offset
                    if el.quarterLength > max_duration:
                        el.quarterLength = max_duration

    # Create final score with chords and melody
    score = stream.Score()
    score.insert(0, melody_measures)
    score.insert(0, chords_part)

    return score
# end overlay_generated_harmony

def save_harmonized_score(score, title="Harmonized Piece", out_path="harmonized.xml"):
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    if out_path.endswith('.xml') or out_path.endswith('.mxl') or out_path.endswith('.musicxml'):
        score.write('musicxml', fp=out_path)
    elif out_path.endswith('.mid') or out_path.endswith('.midi'):
        score.write('midi', fp=out_path)
    else:
        print('uknown file format for file: ', out_path)
# end save_harmonized_score

def load_model(
    curriculum_type='random',
    subfolder=None,
    device_name='cuda:0',
    tokenizer=None,
    pianoroll_dim=100,
    total_stages=10
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = GridMLMMelHarm(
        chord_vocab_size=len(tokenizer.vocab),
        device=device,
        pianoroll_dim=pianoroll_dim,
        max_stages=total_stages
    )
    if curriculum_type == 'random':
        model_path = 'saved_models/' + subfolder + '/' + curriculum_type + str(total_stages) +  '.pt'
    else:
        model_path = 'saved_models/' + subfolder + '/' + curriculum_type +  '.pt'
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_model

def load_model_no_stage(
    curriculum_type='random',
    subfolder='CA',
    device_name='cuda:0',
    tokenizer=None,
    total_stages=10
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = GridMLMMelHarmNoStage(
        chord_vocab_size=len(tokenizer.vocab),
        device=device,
        max_stages=total_stages
    )
    if curriculum_type == 'random':
        model_path = 'saved_models/' + subfolder + '/no_stage/' + curriculum_type + str(total_stages) +  '.pt'
    else:
        model_path = 'saved_models/' + subfolder + '/no_stage/' + curriculum_type + '.pt'
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_model

def generate_files_with_base2(
        model,
        tokenizer,
        input_f,
        mxl_folder,
        midi_folder,
        name_suffix,
        use_constraints=False,
        normalize_tonality=False
    ):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode( input_f, keep_durations=True, normalize_tonality=False )

    harmony_real = torch.LongTensor(input_encoded['input_ids']).reshape(1, len(input_encoded['input_ids']))
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    conditioning_vec = torch.FloatTensor( input_encoded['time_signature'] ).reshape( 1, len(input_encoded['time_signature']) )

    base2_generated_harmony = structured_progressive_generate(
        model=model,
        melody_grid=melody_grid.to(model.device),
        conditioning_vec=conditioning_vec.to(model.device),
        num_stages=10,
        mask_token_id=tokenizer.mask_token_id,
        temperature=1.0,
        strategy='sample',
        pad_token_id=pad_token_id,      # token ID for <pad>
        nc_token_id=nc_token_id,       # token ID for <nc>
        force_fill=True,         # disallow <pad>/<nc> before melody ends
        chord_constraints = harmony_real.to(model.device) if use_constraints else None
    )
    gen_output_tokens = []
    for t in base2_generated_harmony[0].tolist():
        gen_output_tokens.append( tokenizer.ids_to_tokens[t] )
    # keep ground truth
    harmony_real_tokens = []
    for t in harmony_real[0].tolist():
        harmony_real_tokens.append( tokenizer.ids_to_tokens[t] )
    
    gen_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        gen_output_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        gen_score = transpose_score(gen_score, input_encoded['back_interval'])
    mxl_file_name = mxl_folder + f'gen_{name_suffix}' + '.mxl'
    midi_file_name = midi_folder + f'gen_{name_suffix}' + '.mid'
    save_harmonized_score(gen_score, out_path=mxl_file_name)
    os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    real_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        harmony_real_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        real_score = transpose_score(real_score, input_encoded['back_interval'])
    mxl_file_name = mxl_folder + f'real_{name_suffix}' + '.mxl'
    midi_file_name = midi_folder + f'real_{name_suffix}' + '.mid'
    save_harmonized_score(real_score, out_path=mxl_file_name)
    os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score
# end generate_files_with_base2

def generate_files_with_random(
        model,
        tokenizer,
        input_f,
        mxl_folder,
        midi_folder,
        name_suffix,
        use_constraints=False,
        normalize_tonality=False
    ):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode( input_f, keep_durations=True, normalize_tonality=normalize_tonality )

    harmony_real = torch.LongTensor(input_encoded['input_ids']).reshape(1, len(input_encoded['input_ids']))
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    conditioning_vec = torch.FloatTensor( input_encoded['time_signature'] ).reshape( 1, len(input_encoded['time_signature']) )

    random_generated_harmony = random_progressive_generate(
        model=model,
        melody_grid=melody_grid.to(model.device),
        conditioning_vec=conditioning_vec.to(model.device),
        num_stages=10,
        mask_token_id=tokenizer.mask_token_id,
        temperature=1.0,
        strategy='sample',
        pad_token_id=pad_token_id,      # token ID for <pad>
        nc_token_id=nc_token_id,       # token ID for <nc>
        force_fill=True,         # disallow <pad>/<nc> before melody ends
        chord_constraints = harmony_real.to(model.device) if use_constraints else None
    )
    gen_output_tokens = []
    for t in random_generated_harmony[0].tolist():
        gen_output_tokens.append( tokenizer.ids_to_tokens[t] )
    # keep ground truth
    harmony_real_tokens = []
    for t in harmony_real[0].tolist():
        harmony_real_tokens.append( tokenizer.ids_to_tokens[t] )
    
    gen_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        gen_output_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        gen_score = transpose_score(gen_score, input_encoded['back_interval'])
    mxl_file_name = mxl_folder + f'gen_{name_suffix}' + '.mxl'
    midi_file_name = midi_folder + f'gen_{name_suffix}' + '.mid'
    save_harmonized_score(gen_score, out_path=mxl_file_name)
    os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    real_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        harmony_real_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        real_score = transpose_score(real_score, input_encoded['back_interval'])
    mxl_file_name = mxl_folder + f'real_{name_suffix}' + '.mxl'
    midi_file_name = midi_folder + f'real_{name_suffix}' + '.mid'
    save_harmonized_score(real_score, out_path=mxl_file_name)
    os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score
# end generate_files_with_random