import torch
import torch.nn.functional as F
from train_utils import apply_structured_masking
from music21 import harmony, stream, metadata, chord, meter
import mir_eval
import numpy as np
from copy import deepcopy
from models import GridMLMMelHarm

def random_progressive_generate(
    model,
    melody_grid,            # (1, seq_len, input_dim)
    conditioning_vec,       # (1, cond_dim)
    num_stages,             # e.g., 10
    mask_token_id,          # token ID used for masking
    temperature=1.0,        # optional temperature for sampling
    strategy='topk',        # 'topk' or 'sample' strategy for selecting new tokens
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]
    tokens_per_stage = int(seq_len / num_stages + 0.5)

    # Start with all tokens masked
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    for stage in range(num_stages):
        with torch.no_grad():
            logits = model(
                melody_grid=melody_grid.to(model.device),
                conditioning_vec=conditioning_vec.to(model.device),
                harmony_tokens=visible_harmony.to(model.device),
                stage_indices=torch.LongTensor([stage]).to(model.device)
            )  # (1, seq_len, vocab_size)

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
            selected_positions = masked_positions[torch.multinomial(p, topk, replacement=False)]
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
    strategy='topk'         # 'topk' or 'sample'
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

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
    chords_part = stream.Part()
    # Create deep copy of flat melody part
    harmonized_part = deepcopy(melody_part)
    
    # Remove old chord symbols
    for el in harmonized_part.recurse().getElementsByClass(harmony.ChordSymbol):
        harmonized_part.remove(el)

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

        # harmonized_part.insert(offset, chord_symbol)
        chords_part.insert(offset, c)
        inserted_chords[i] = chord_symbol
        last_chord_symbol = mir_chord

    # Convert flat part to one with measures
    harmonized_with_measures = harmonized_part.makeMeasures()

    # Repeat previous chord at start of bars with no chord
    for m in harmonized_with_measures.getElementsByClass(stream.Measure):
        bar_offset = m.offset
        # has_chord = any(isinstance(el, harmony.ChordSymbol) and el.offset == bar_offset for el in m)
        # has_chord = any( isinstance(el, harmony.ChordSymbol) for el in m )
        has_chord = any(isinstance(el, harmony.ChordSymbol) and el.offset == 0. for el in m)
        if not has_chord:
            # Find previous chord before this measure
            prev_chords = [el for el in harmonized_part.recurse().getElementsByClass(harmony.ChordSymbol)
                           if el.offset < bar_offset]
            if prev_chords:
                prev_chord = prev_chords[-1]
                m.insert(0.0, deepcopy(prev_chord))
    
    # Convert flat part to one with measures
    chords_with_measures = chords_part.makeMeasures()

    # Repeat previous chord at start of bars with no chord
    for m in chords_with_measures.getElementsByClass(stream.Measure):
        bar_offset = m.offset
        # has_chord = any(isinstance(el, chord.Chord) and el.offset == bar_offset for el in m)
        # has_chord = any( isinstance(el, chord.Chord) for el in m )
        has_chord = any(isinstance(el, chord.Chord) and el.offset == 0. for el in m)
        if not has_chord:
            # Find previous chord before this measure
            prev_chords = [el for el in chords_part.recurse().getElementsByClass(chord.Chord)
                           if el.offset < bar_offset]
            if prev_chords:
                prev_chord = prev_chords[-1]
                m.insert(0.0, deepcopy(prev_chord))

    # Create final score with chords and melody
    score = stream.Score()
    score.insert(0, harmonized_with_measures)
    score.insert(0, chords_with_measures)

    return score
# end overlay_generated_harmony

def save_harmonized_score(score, title="Harmonized Piece", out_path="harmonized.xml"):
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.write('musicxml', fp=out_path)
# end save_harmonized_score

def load_model(curriculum_type = 'random', device_name = 'cuda:0', tokenizer=None):
    curriculum_type = 'random'
    device_name = 'cuda:1'
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    model = GridMLMMelHarm(
        chord_vocab_size=len(tokenizer.vocab),
        device=device
    )
    model_path = 'saved_models/' + curriculum_type +  '.pt'
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_model