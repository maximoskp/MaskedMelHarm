import torch
import torch.nn.functional as F
from train_utils import apply_structured_masking
from music21 import harmony, stream, metadata, chord, note, key, meter, tempo, duration, clef
import mir_eval
import numpy as np
from copy import deepcopy
from models import GridMLMMelHarm, GridMLMMelHarmNoStage, DualGridMLMMelHarm, SingleGridMLMelHarm, SEModular
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


def greedy_token_by_token_generate(
    model,
    melody_grid,            # (1, seq_len, input_dim)
    conditioning_vec,       # (1, cond_dim)
    num_stages,             # e.g., 10
    mask_token_id,          # token ID used for masking
    bar_token_id,           # token ID for bar markers
    temperature=1.0,        # optional softmax temperature
    pad_token_id=None,      # token ID for <pad>
    nc_token_id=None,       # token ID for <nc>
    force_fill=True,        # disallow <pad>/<nc> before melody ends
    chord_constraints=None, # chord + bar constraints
    max_steps=None          # optional limit on number of iterations
):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    # --- 1. Initialize ---
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id,
                                 chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]

    # Compute last active melody index if forcing fill
    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)  # shape: (seq_len,)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except:
            last_active_index = -1
    else:
        last_active_index = -1

    # Prepare for convergence tracking
    prev_logits = None
    avg_diffs = []

    # --- 2. Iterative greedy unmasking ---
    total_tokens = visible_harmony.numel()
    step = 0
    while (visible_harmony == mask_token_id).any():
        if max_steps is not None and step >= max_steps:
            break
        num_masked = (visible_harmony == mask_token_id).sum().item()
        num_unmasked = total_tokens - num_masked
        s = max(round((num_unmasked / total_tokens) * num_stages)-1, 0)
        with torch.no_grad():
            logits = model(
                melody_grid=melody_grid.to(model.device),
                conditioning_vec=conditioning_vec.to(model.device),
                harmony_tokens=visible_harmony.to(model.device),
                stage_indices=torch.LongTensor([s]).to(model.device)  # optional if model expects stage
            )  # (1, seq_len, vocab_size)

        # Mask out invalid predictions if enforcing force_fill
        if force_fill and (pad_token_id is not None and nc_token_id is not None):
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float('-inf')
                    logits[0, i, nc_token_id] = float('-inf')
                else:
                    logits[0, i, :] = float('-inf')
                    logits[0, i, pad_token_id] = 1.0

        # Compute probabilities
        probs = torch.softmax(logits / temperature, dim=-1)  # (1, seq_len, vocab_size)
        confidences, predictions = torch.max(probs, dim=-1)  # (1, seq_len)

        # --- Convergence metric ---
        if prev_logits is not None:
            # Only compare masked positions
            masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
            if masked_positions.numel() > 0:
                prev_p = torch.softmax(prev_logits[0, masked_positions] / temperature, dim=-1)
                curr_p = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
                # Mean absolute difference across vocab dimension, then average over positions
                mad = torch.mean(torch.abs(prev_p - curr_p)).item()
                avg_diffs.append(mad)
        prev_logits = logits.clone()


        # # --- Greedy pick: unmask token with highest confidence ---
        # masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        # if masked_positions.numel() == 0:
        #     break

        # masked_confidences = confidences[0, masked_positions]
        # best_idx = masked_positions[torch.argmax(masked_confidences)].item()
        # visible_harmony[0, best_idx] = predictions[0, best_idx]

        # --- Sampling pick: unmask one token using top-k temperature sampling ---
        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break

        # Get logits only at masked positions
        masked_logits = logits[0, masked_positions] / temperature  # (num_masked, vocab_size)

        # For each masked position, find the most confident one (like greedy),
        # but sample the actual token value with top-k sampling
        masked_probs = torch.softmax(masked_logits, dim=-1)
        masked_confidences, _ = torch.max(masked_probs, dim=-1)

        # Pick the masked position with highest confidence
        best_pos_idx = torch.argmax(masked_confidences).item()
        pos = masked_positions[best_pos_idx].item()

        # --- Top-k sampling at this position ---
        k = min(10, masked_logits.size(-1))  # choose k (adjustable)
        topk_logits, topk_indices = torch.topk(masked_logits[best_pos_idx], k)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        sampled_idx = torch.multinomial(topk_probs, 1).item()
        sampled_token = topk_indices[sampled_idx].item()

        # Update visible_harmony with the sampled token
        visible_harmony[0, pos] = sampled_token

        step += 1

    return visible_harmony, avg_diffs
# end greedy_token_by_token_generate

def beam_token_by_token_generate(
        model,
        melody_grid,            # (1, seq_len, input_dim)
        mask_token_id,          # token ID used for masking
        bar_token_id,           # token ID for bar markers
        temperature=1.0,        # optional softmax temperature
        pad_token_id=None,      # token ID for <pad>
        nc_token_id=None,       # token ID for <nc>
        force_fill=True,        # disallow <pad>/<nc> before melody ends
        chord_constraints=None, # chord + bar constraints
        beam_size=5,            # number of beams to keep
        top_k=5,                # number of candidates per expansion
        unmasking_order='random', # in ['random', 'start', 'end', 'certain', 'uncertain']
    ):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    # --- 1. Initialize ---
    init_visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id,
                                 chord_constraints != pad_token_id)
        init_visible_harmony[idxs] = chord_constraints[idxs]
    
    # Compute last active melody index if forcing fill
    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)  # shape: (seq_len,)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except:
            last_active_index = -1
    else:
        last_active_index = -1

    # Each beam is (visible_harmony, score, avg_diffs, prev_logits)
    beams = [(init_visible_harmony.clone(), 0.0, [], None)]
    total_tokens = init_visible_harmony.numel()
    
    step = 0
    while any((bh[0] == mask_token_id).any() for bh in beams):
        candidates = []
        # print('entering beams==========================================')
        for visible_harmony, score, avg_diffs, prev_logits in beams:

            with torch.no_grad():
                logits1 = model(
                    melody_grid=melody_grid.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    # stage_indices=torch.LongTensor([s]).to(model.device)
                )  # (1, seq_len, vocab_size)
                logits = logits1

            # --- Masked position selection ---
            masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
            if masked_positions.numel() == 0:
                candidates.append((visible_harmony, score, avg_diffs, logits.clone()))
                continue
            # Compute entropies over masked positions
            probs = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
            masked_probs = probs  # (num_masked, vocab_size)
            entropies = -(masked_probs * masked_probs.clamp_min(1e-9).log()).sum(dim=-1)

            if unmasking_order == 'random':
                # pick position at random
                pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()
            elif unmasking_order == 'uncertain':
                # Pick position with highest entropy
                pos = masked_positions[torch.argmax(entropies)].item()
            elif unmasking_order == 'certain':
                # Pick position with lowest entropy
                pos = masked_positions[torch.argmin(entropies)].item()
            elif unmasking_order == 'start':
                # pick position from start to end
                pos = masked_positions[0].item()
            elif unmasking_order == 'end':
                # pick position from end to start
                pos = masked_positions[-1].item()
            else:
                print('Unknown unmasking order: ', unmasking_order, '. Doing random.')
                # pick position at random
                pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()

            # Mask out invalid predictions if enforcing force_fill
            if force_fill and (pad_token_id is not None and nc_token_id is not None):
                for i in range(seq_len):
                    if i <= last_active_index:
                        logits[0, i, pad_token_id] = float('-inf')
                        logits[0, i, nc_token_id] = float('-inf')
                    else:
                        logits[0, i, :] = float('-inf')
                        logits[0, i, pad_token_id] = 1.0

            # --- Convergence metric ---
            if prev_logits is not None:
                masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
                if masked_positions.numel() > 0:
                    prev_p = torch.softmax(prev_logits[0, masked_positions] / temperature, dim=-1)
                    curr_p = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
                    mad = torch.mean(torch.abs(prev_p - curr_p)).item()
                    avg_diffs = avg_diffs + [mad]

            # --- Top-k sampling expansion ---
            masked_logits = logits[0, pos] / temperature
            topk_logits, topk_indices = torch.topk(masked_logits, min(top_k, masked_logits.size(-1)))
            topk_probs = torch.softmax(topk_logits, dim=-1)
            # print('---------------- pos: ', pos)
            # print(visible_harmony)
            # print(topk_indices)
            # print(topk_probs)
            for j in range(topk_indices.size(0)):
                token = topk_indices[j].item()
                token_prob = topk_probs[j].item()

                new_harmony = visible_harmony.clone()
                new_harmony[0, pos] = token
                new_score = score + torch.log(torch.tensor(token_prob + 1e-12)).item()

                candidates.append((new_harmony, new_score, avg_diffs, logits.clone()))

        # Keep top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        step += 1

    # Return best beam (highest score)
    best_harmony, _, best_avg_diffs, _ = beams[0]
    return best_harmony, best_avg_diffs
# end beam_token_by_token_generate

def nucleus_token_by_token_generate(
        model,
        melody_grid,            # (1, seq_len, input_dim)
        mask_token_id,          # token ID used for masking
        temperature=1.0,        # optional softmax temperature
        pad_token_id=None,      # token ID for <pad>
        nc_token_id=None,       # token ID for <nc>
        force_fill=True,        # disallow <pad>/<nc> before melody ends
        chord_constraints=None, # chord + bar constraints
        p=0.9,                  # nucleus threshold
        unmasking_order='random', # in ['random', 'start', 'end', 'certain', 'uncertain']
        num_stages=None,
        conditioning_vec=None
    ):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    # --- 1. Initialize ---
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id,
                                 chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]
    # Compute last active melody index if forcing fill
    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)  # shape: (seq_len,)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except:
            last_active_index = -1
    else:
        last_active_index = -1

    step = 0
    while (visible_harmony == mask_token_id).any():
        with torch.no_grad():
            stage = int(
                ((visible_harmony == mask_token_id).sum().item()/visible_harmony.size().numel())*num_stages
            )
            if conditioning_vec is None:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device)
                )  # (1, seq_len, vocab_size)
            else:
                logits = model(
                    melody_grid=melody_grid.to(model.device),
                    conditioning_vec=conditioning_vec.to(model.device),
                    harmony_tokens=visible_harmony.to(model.device),
                    stage_indices=torch.LongTensor([stage]).to(model.device)
                )  # (1, seq_len, vocab_size)
        # --- Masked position selection ---
        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break

        probs = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
        entropies = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)

        if unmasking_order == 'random':
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()
        elif unmasking_order == 'uncertain':
            pos = masked_positions[torch.argmax(entropies)].item()
        elif unmasking_order == 'certain':
            pos = masked_positions[torch.argmin(entropies)].item()
        elif unmasking_order == 'start':
            pos = masked_positions[0].item()
        elif unmasking_order == 'end':
            pos = masked_positions[-1].item()
        else:
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()

        # Mask out invalid predictions if enforcing force_fill
        if force_fill and (pad_token_id is not None and nc_token_id is not None):
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float('-inf')
                    logits[0, i, nc_token_id] = float('-inf')
                else:
                    logits[0, i, :] = float('-inf')
                    logits[0, i, pad_token_id] = 1.0

        # --- Nucleus sampling step ---
        logits_pos = logits[0, pos] / temperature
        probs_pos = torch.softmax(logits_pos, dim=-1)

        # sort probs descending
        sorted_probs, sorted_idx = torch.sort(probs_pos, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask out tokens beyond nucleus p
        nucleus_mask = cumulative_probs <= p
        nucleus_mask[0] = True  # keep at least one token
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_idx = sorted_idx[nucleus_mask]

        # renormalize
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        # sample
        sampled_idx = torch.multinomial(nucleus_probs, 1).item()
        token = nucleus_idx[sampled_idx].item()

        # update harmony
        visible_harmony[0, pos] = token
        step += 1
    
    return visible_harmony
# end nucleus_token_by_token_generate

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


def _typed_duration(ql):

    d = duration.Duration(ql)
    base = {4.0:'whole', 2.0:'half', 1.0:'quarter', 0.5:'eighth', 0.25:'16th', 0.125:'32nd'}
    for dots in (0,1,2):
        unit = ql / (2**dots)
        if unit in base:
            d.type = base[unit]
            d.dots = dots
            break
    return d

def _decode_token_to_chord(token):
    # 'C:maj7' -> triad/7th (reduced extensions) as a Chord (absolute MIDI)
    try:
        r, t, _ = mir_eval.chord.encode(token, reduce_extended_chords=True)
        pcs = r + np.where(t > 0)[0] + 48  # anchor roughly around C3
        return chord.Chord(pcs.tolist())
    except Exception:
        return None

def _voice_into_register(c: chord.Chord,
                         low=45, high=65,   # A2–F4 default register for a LH staff
                         last_bass: int | None = None) -> chord.Chord:
    """
    Put chord into closed position and choose an octave placement (k*12)
    that keeps it inside [low, high] and is as close as possible to the
    previous chord’s bass (if any).
    """
    # closed stack within one octave
    c = chord.Chord([p.midi for p in c.pitches])
    c.closedPosition(inPlace=True)

    def span(ch):
        mids = [p.midi for p in ch.pitches]
        return min(mids), max(mids)

    best = None
    bestDist = None
    # try several octave shifts and pick the closest, in range
    for k in range(-3, 4):
        cand = c.transpose(12 * k, inPlace=False)
        lo, hi = span(cand)
        if lo < low or hi > high:
            continue
        bass = lo
        dist = 0 if last_bass is None else abs(bass - last_bass)
        if best is None or dist < bestDist:
            best, bestDist = cand, dist

    # if nothing fit, just clamp roughly by dragging into range
    if best is None:
        lo, hi = span(c)
        while lo < low:
            c = c.transpose(12, inPlace=False)
            lo, hi = span(c)
        while hi > high:
            c = c.transpose(-12, inPlace=False)
            lo, hi = span(c)
        best = c

    return best

def _force_bass_clef(part: stream.Part):
    """
    Make sure the part shows a Bass clef at the start.
    - Remove ALL existing clefs (including mid-score changes).
    - Ensure measures exist.
    - Insert a Bass clef in MEASURE 1 at offset 0.
    """
    # remove all clefs
    for c in list(part.recurse().getElementsByClass(clef.Clef)):
        parent = c.getContextByClass(stream.Measure)
        (parent or part).remove(c)

    # ensure we have measures
    if not part.getElementsByClass(stream.Measure):
        part.makeMeasures(inPlace=True)

    m1 = part.measure(1)
    if m1 is None:  # very defensive
        part.makeMeasures(inPlace=True)
        m1 = part.measure(1)

    # insert the clef INSIDE measure 1 at 0.0
    m1.insert(0.0, clef.BassClef())

def _apply_keyed_accidentals(score: stream.Score) -> stream.Score:
    """
    Normalize enharmonics toward the part's key and recompute accidental display.
    Works on both Note and Chord elements. Compatible with older music21 versions.
    """
    for p in score.parts:
        # 1) ensure a KeySignature exists (infer if missing)
        ks_stream = p.recurse().getElementsByClass(key.KeySignature)
        if not ks_stream:
            try:
                inferred = p.analyze('key').asKeySignature()
                p.insert(0, inferred)
                ks_stream = [inferred]
            except Exception:
                ks_stream = []

        # 2) respell enharmonics roughly toward the key's sharp/flat preference
        prefer_sharps = True
        if ks_stream:
            prefer_sharps = ks_stream[0].sharps >= 0  # flats if negative

        # visit both Note and Chord; if Chord, visit its member notes
        for el in p.recurse().getElementsByClass((note.Note, chord.Chord)):
            if isinstance(el, note.Note):
                notes_iter = [el]
            else:  # Chord
                notes_iter = el.notes  # member Note objects

            for n in notes_iter:
                acc = n.pitch.accidental
                if acc is None:
                    continue
                # steer enharmonic toward the key's preference
                if prefer_sharps and acc.alter < 0:
                    n.pitch = n.pitch.getEnharmonic()
                elif not prefer_sharps and acc.alter > 0:
                    n.pitch = n.pitch.getEnharmonic()

        # 3) recompute accidental DISPLAY (try newest → oldest signatures)
        try:
            p.makeAccidentals(inPlace=True, overrideStatus=True, cautionaryAccidentals=False)
        except TypeError:
            try:
                p.makeAccidentals(inPlace=True)
            except TypeError:
                p.makeAccidentals()

    return score


def overlay_generated_harmony_old_1(melody_part, generated_chords, ql_per_16th, skip_steps):
    """
    Two-part score:
      • melody_measures: original melody in measures
      • chords_part: bar-aligned structure with voiced chords of full duration
    If the melody is longer than the generated sequence, the score is cropped
    to the last bar touched by the generated sequence.
    """
    EPS = 1e-9

    # 1) measured copy of the melody
    melody_measures = melody_part.makeMeasures(inPlace=False)

    # 2) bar skeleton for chords (keep clef/time/key/tempo only)
    chords_part = deepcopy(melody_measures)
    chords_part.id = 'Harmonies'
    for m in chords_part.getElementsByClass(stream.Measure):
        for el in list(m):
            if not isinstance(el, (key.KeySignature, meter.TimeSignature, tempo.MetronomeMark, clef.Clef)):
                m.remove(el)

    # 2) Force Bass Cleff
    _force_bass_clef(chords_part)

    # measure boundaries (absolute)
    measures = list(chords_part.getElementsByClass(stream.Measure))
    boundaries = [(m.offset, m.offset + m.barDuration.quarterLength, m) for m in measures]
    if not boundaries:
        sc = stream.Score()
        sc.insert(0, melody_measures)
        sc.insert(0, chords_part)
        return sc

    # 3) group contiguous runs of identical, meaningful tokens
    tokens = list(generated_chords)
    N = len(tokens)
    runs, i = [], 0
    while i < N:
        tok = tokens[i]
        if tok in ('<pad>', '<nc>'):
            i += 1; continue
        j = i + 1
        while j < N and tokens[j] == tok:
            j += 1
        runs.append((tok, i, j))  # [i, j)
        i = j

    # 4) insert chords with full durations (clamped per bar) + register voicing
    last_bass = None
    for tok, i0, i1 in runs:
        start_t = (i0 + skip_steps) * ql_per_16th
        end_t   = (i1 + skip_steps) * ql_per_16th
        if end_t <= start_t + EPS:
            continue

        proto = _decode_token_to_chord(tok)
        if proto is None:
            continue

        # choose a voiced version for this run (near last bass)
        voiced_template = _voice_into_register(proto, low=45, high=65, last_bass=last_bass)
        last_bass = min(p.midi for p in voiced_template.pitches)

        for b_start, b_end, meas in boundaries:
            if b_end <= start_t or b_start >= end_t:
                continue
            seg_start = max(start_t, b_start)
            seg_end   = min(end_t,   b_end)
            seg_dur   = max(0.0, seg_end - seg_start)
            if seg_dur <= EPS:
                continue

            # copy voiced template and set the segment duration
            c = chord.Chord([p.midi for p in voiced_template.pitches])
            c.duration = _typed_duration(seg_dur)
            meas.insert(seg_start - b_start, c)

    # 5) assemble result
    score = stream.Score()
    score.insert(0, melody_measures)
    score.insert(0, chords_part)

    # 6) crop to the last bar covered by generated sequence
    if N > 0:
        seq_end = (skip_steps + N) * ql_per_16th
        last_idx = -1
        for idx, (bs, _, _) in enumerate(boundaries):
            if bs < seq_end - EPS:
                last_idx = idx
        if last_idx >= 0:
            last_bar = measures[last_idx].measureNumber
            score = score.measures(1, last_bar)

    # 7) Fix accidentals
    _apply_keyed_accidentals(score)

    return score



def overlay_generated_harmony_old_0(melody_part, generated_chords, ql_per_16th, skip_steps):
    """
    Build a two‐part Score:
      • melody_measures: your original melody, in measures
      • chords_part:   identical bar structure but empty, into which we
                      insert one chord per bar (no ties)
    """

    # 1) Make your melody into a measured stream
    melody_measures = melody_part.makeMeasures(inPlace=False)

    # 2) Deep-copy that structure for harmony
    chords_part = deepcopy(melody_measures)
    chords_part.id = 'Harmonies'

    # 2a) Strip out _all_ notes/chords from the copy, leaving only bars & sigs
    for m in chords_part.getElementsByClass(stream.Measure):
        for el in list(m):
            if not isinstance(el, (key.KeySignature,
                                   meter.TimeSignature,
                                   tempo.MetronomeMark,
                                   clef.Clef            # <-- keep any clef
                                   )):
                m.remove(el)

    # 2b) Copy initial Key/Time/Tempo markings into chords_part
    for el in melody_measures.recurse().getElementsByClass((key.KeySignature,
                                                            meter.TimeSignature,
                                                            tempo.MetronomeMark,
                                                            clef.Clef)):
        if el.offset == 0:
            chords_part.insert(0, el)

    # 3) Recompute offsets on melody_measures
    cumulative = 0.0
    measures_m = list(melody_measures.getElementsByClass(stream.Measure))
    for m in measures_m:
        m.offset = cumulative
        cumulative += m.barDuration.quarterLength

    #  ─── Build a [start,end,measure] list, but grab the measure _from_ chords_part ───
    measures_c = list(chords_part.getElementsByClass(stream.Measure))
    measure_boundaries = []
    for m_m, m_c in zip(measures_m, measures_c):
        start = m_m.offset
        end   = start + m_m.barDuration.quarterLength
        measure_boundaries.append((start, end, m_c))

    # 4) Walk through generated_chords, group repeats, and insert one chord‐per‐bar
    step = 0
    N    = len(generated_chords)
    while step < N:
        token = generated_chords[step]
        # skip <pad> and <nc>
        if token in ('<pad>','<nc>'):
            step += 1
            continue

        # find run of the same token
        start = step
        while step < N and generated_chords[step] == token:
            step += 1
        end = step

        region_start = (start + skip_steps) * ql_per_16th
        region_end   = (end   + skip_steps) * ql_per_16th

        # insert into every bar that overlaps this region
        for bar_start, bar_end, measure in measure_boundaries:
            if bar_end <= region_start or bar_start >= region_end:
                continue

            seg_start  = max(region_start, bar_start)
            seg_end    = min(region_end,   bar_end)
            seg_dur    = seg_end - seg_start
            seg_offset = seg_start - bar_start

            # decode chord symbol → Chord object
            try:
                r, t, _ = mir_eval.chord.encode(token,
                                                reduce_extended_chords=True)
                pcs = r + np.where(t > 0)[0] + 48
                c   = chord.Chord(pcs.tolist())
            except Exception as e:
                print(f"Skipping invalid chord {token}: {e}")
                continue

            c.quarterLength = seg_dur
            measure.insert(seg_offset, c)

    # 5) Assemble final Score
    score = stream.Score()
    score.insert(0, melody_measures)
    score.insert(0, chords_part)
    return score


def overlay_generated_harmony(melody_part, generated_chords, ql_per_16th, skip_steps):
    # create a part for chords in midi format
    # melody_part = melody_part.makeMeasures()
    # chords_part = deepcopy(melody_part)
    # Create deep copy of flat melody part
    # Create a new part for filtered content
    filtered_part = stream.Part()
    # filtered_part.id = melody_part.id  # Preserve ID

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

    # keep bar tokens out of the steps count
    num_bar_tokens = 0
    for i, mir_chord in enumerate(generated_chords):
        if mir_chord in ("<pad>", "<nc>"):
            continue
        if mir_chord == last_chord_symbol:
            continue
        if mir_chord == "<bar>":
            num_bar_tokens += 1
            continue
        
        offset = (i + skip_steps - num_bar_tokens) * ql_per_16th
        
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

        offset = (i + skip_steps - num_bar_tokens) * ql_per_16th
        
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

def load_DE_model(
    d_model=512, 
    nhead=4, 
    num_layers_mel=4,
    num_layers_harm=4,
    curriculum_type='f2f',
    subfolder=None,
    device_name='cuda:0',
    tokenizer=None,
    melody_length=80,
    harmony_length=80,
    exponent=5,
    nvis=None
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = DualGridMLMMelHarm(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers_mel=num_layers_mel,
        num_layers_harm=num_layers_harm,
        device=device,
        melody_length=melody_length,
        harmony_length=harmony_length,
        pianoroll_dim=tokenizer.pianoroll_dim,
    )
    model_path = 'saved_models/' + subfolder + '/' + curriculum_type + str(exponent)
    if nvis is not None:
        model_path += '_nvis' + str(nvis)
    model_path += '.pt'
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_DE_model

def load_SE_model(
    d_model=512, 
    nhead=4, 
    num_layers=4,
    curriculum_type='f2f',
    subfolder=None,
    device_name='cuda:0',
    tokenizer=None,
    grid_length=80,
    exponent=5,
    nvis=None,
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = SingleGridMLMelHarm(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        device=device,
        grid_length=grid_length,
        pianoroll_dim=tokenizer.pianoroll_dim,
    )
    model_path = 'saved_models/' + subfolder + '/' + curriculum_type + str(exponent)
    if nvis is not None:
        model_path += '_nvis' + str(nvis)
    model_path += '.pt'
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_SE_model

def load_SE_Modular(
    d_model=512, 
    nhead=8, 
    num_layers=8,
    curriculum_type='f2f',
    subfolder=None,
    device_name='cuda:0',
    tokenizer=None,
    grid_length=80,
    nvis=None,
    condition_dim=None,  # if not None, add a condition token of this dim at start
    unmasking_stages=None,  # if not None, use stage-based unmasking
    trainable_pos_emb=False,
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = SEModular(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        device=device,
        grid_length=grid_length,
        pianoroll_dim=tokenizer.pianoroll_dim,
        condition_dim=condition_dim,  # if not None, add a condition token of this dim at start
        unmasking_stages=unmasking_stages,  # if not None, use stage-based unmasking
        trainable_pos_emb=trainable_pos_emb
    )
    model_path = 'saved_models/SE/' + subfolder + '/' + curriculum_type
    if nvis is not None:
        model_path += '_nvis' + str(nvis)
    model_path += '.pt'
    print('model_path: ',model_path)
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_SE_Modular

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
        normalize_tonality=False,
        num_stages=10
    ):
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode( input_f, keep_durations=True, normalize_tonality=normalize_tonality )

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

    # ─────── DEBUG PRINT ───────────────────────────────────────────
    print("Generated harmony tokens:")
    print(gen_output_tokens)
    print("Ground-truth harmony tokens:")
    print(harmony_real_tokens)
    # ────────────────────────────────────────────────────────────────
    
    gen_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        gen_output_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        gen_score = transpose_score(gen_score, input_encoded['back_interval'])
    mxl_file_name = os.path.join(mxl_folder, f'gen_{name_suffix}' + '.mxl')
    midi_file_name = os.path.join(midi_folder, f'gen_{name_suffix}' + '.mid')
    save_harmonized_score(gen_score, out_path=mxl_file_name)
    save_harmonized_score(gen_score, out_path=midi_file_name)
    # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    real_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        harmony_real_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        real_score = transpose_score(real_score, input_encoded['back_interval'])
    mxl_file_name = os.path.join(mxl_folder, f'real_{name_suffix}' + '.mxl')
    midi_file_name = os.path.join(midi_folder, f'real_{name_suffix}' + '.mid')
    save_harmonized_score(real_score, out_path=mxl_file_name)
    save_harmonized_score(real_score, out_path=midi_file_name)
    # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

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
        normalize_tonality=False,
        num_stages=10
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
        num_stages=num_stages,
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
    mxl_file_name = os.path.join(mxl_folder, f'gen_{name_suffix}' + '.mxl')
    midi_file_name = os.path.join(midi_folder, f'gen_{name_suffix}' + '.mid')
    save_harmonized_score(gen_score, out_path=mxl_file_name)
    save_harmonized_score(gen_score, out_path=midi_file_name)
    # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    real_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        harmony_real_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    if normalize_tonality:
        real_score = transpose_score(real_score, input_encoded['back_interval'])
    mxl_file_name = os.path.join(mxl_folder, f'real_{name_suffix}' + '.mxl')
    midi_file_name = os.path.join(midi_folder, f'real_{name_suffix}' + '.mid')
    save_harmonized_score(real_score, out_path=mxl_file_name)
    save_harmonized_score(real_score, out_path=midi_file_name)
    # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score
# end generate_files_with_random

def generate_files_with_greedy(
        model,
        tokenizer,
        input_f,
        mxl_folder,
        midi_folder,
        name_suffix,
        use_constraints=False,
        condition='time_signature',
        force_condition=None,
        intertwine_bar_info=False, # no bar default
        trim_start=True, # no bar default
        normalize_tonality=False,
        num_stages=10,
        temperature=1.0
    ):
    # we cannot have intertwine_bar_info == True and use_constraints == False
    # because bar information is passed through the constraints
    # if intertwine_bar_info:
    #     use_constraints = True

    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
            input_f,
            keep_durations=True,
            normalize_tonality=normalize_tonality,
        )

    harmony_real = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    harmony_input = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    # if intertwine_bar_info is True and use_constraints is False, we only need to pass
    # the bar information as a constraint, not the chords, or anything else
    # so mask out everything except from bar_token_ids
    if intertwine_bar_info and not use_constraints:
        harmony_input[ harmony_input != tokenizer.bar_token_id ] = tokenizer.mask_token_id
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    conditioning_vec = torch.FloatTensor( input_encoded[condition] ).reshape( 1, len(input_encoded[condition]) )
    if force_condition is not None:
        conditioning_vec = torch.FloatTensor( force_condition ).reshape( 1, len(force_condition) )
    
    random_generated_harmony, avg_diffs = greedy_token_by_token_generate(
        model=model,
        melody_grid=melody_grid.to(model.device),
        conditioning_vec=conditioning_vec.to(model.device),
        num_stages=num_stages,
        mask_token_id=tokenizer.mask_token_id,
        bar_token_id=tokenizer.bar_token_id,
        temperature=temperature,
        pad_token_id=pad_token_id,      # token ID for <pad>
        nc_token_id=nc_token_id,       # token ID for <nc>
        force_fill=True,         # disallow <pad>/<nc> before melody ends
        chord_constraints = harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None
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
    mxl_file_name = os.path.join(mxl_folder, f'gen_{name_suffix}' + '.mxl')
    midi_file_name = os.path.join(midi_folder, f'gen_{name_suffix}' + '.mid')
    save_harmonized_score(gen_score, out_path=mxl_file_name)
    save_harmonized_score(gen_score, out_path=midi_file_name)
    # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    real_score = overlay_generated_harmony(
        input_encoded['melody_part'],
        harmony_real_tokens,
        input_encoded['ql_per_quantum'],
        input_encoded['skip_steps']
    )
    
    if normalize_tonality:
        real_score = transpose_score(real_score, input_encoded['back_interval'])
    mxl_file_name = os.path.join(mxl_folder, f'real_{name_suffix}' + '.mxl')
    midi_file_name = os.path.join(midi_folder, f'real_{name_suffix}' + '.mid')
    save_harmonized_score(real_score, out_path=mxl_file_name)
    save_harmonized_score(real_score, out_path=midi_file_name)
    # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score, avg_diffs
# end generate_files_with_greedy

def generate_files_with_beam(
        model,
        tokenizer,
        input_f,
        mxl_folder,
        midi_folder,
        name_suffix,
        use_constraints=False,
        intertwine_bar_info=False, # no bar default
        normalize_tonality=False,
        temperature=1.0,
        beam_size=5,
        top_k=5,
        unmasking_order='random',
        create_gen=True,
        create_real=False
    ):
    # we cannot have intertwine_bar_info == True and use_constraints == False
    # because bar information is passed through the constraints
    # if intertwine_bar_info:
    #     use_constraints = True

    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
            input_f,
            keep_durations=True,
            normalize_tonality=normalize_tonality,
        )

    harmony_real = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    harmony_input = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    # if intertwine_bar_info is True and use_constraints is False, we only need to pass
    # the bar information as a constraint, not the chords, or anything else
    # so mask out everything except from bar_token_ids
    if intertwine_bar_info and not use_constraints:
        harmony_input[ harmony_input != tokenizer.bar_token_id ] = tokenizer.mask_token_id
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    if create_gen:
        random_generated_harmony, avg_diffs = beam_token_by_token_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            mask_token_id=tokenizer.mask_token_id,
            bar_token_id=tokenizer.bar_token_id,
            temperature=temperature,
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True,         # disallow <pad>/<nc> before melody ends
            chord_constraints = harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
            beam_size=beam_size,
            top_k=top_k,
            unmasking_order=unmasking_order,
        )
        gen_output_tokens = []
        for t in random_generated_harmony[0].tolist():
            gen_output_tokens.append( tokenizer.ids_to_tokens[t] )
    else:
        avg_diffs = None
        gen_output_tokens = None
    # keep ground truth
    harmony_real_tokens = []
    for t in harmony_real[0].tolist():
        harmony_real_tokens.append( tokenizer.ids_to_tokens[t] )
    gen_score = None
    real_score = None
    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded['melody_part'],
            gen_output_tokens,
            input_encoded['ql_per_quantum'],
            input_encoded['skip_steps']
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded['back_interval'])
        if mxl_folder is not None:
            mxl_file_name = os.path.join(mxl_folder, f'gen_{name_suffix}' + '.mxl')
            save_harmonized_score(gen_score, out_path=mxl_file_name)
        if midi_folder is not None:
            midi_file_name = os.path.join(midi_folder, f'gen_{name_suffix}' + '.mid')
            save_harmonized_score(gen_score, out_path=midi_file_name)
        # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')
    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded['melody_part'],
            harmony_real_tokens,
            input_encoded['ql_per_quantum'],
            input_encoded['skip_steps']
        )
        
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded['back_interval'])
        if mxl_folder is not None:
            mxl_file_name = os.path.join(mxl_folder, f'real_{name_suffix}' + '.mxl')
            save_harmonized_score(real_score, out_path=mxl_file_name)
        if midi_folder is not None:
            midi_file_name = os.path.join(midi_folder, f'real_{name_suffix}' + '.mid')
            save_harmonized_score(real_score, out_path=midi_file_name)
        # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score, avg_diffs
# end generate_files_with_beam

def generate_files_with_nucleus(
        model,
        tokenizer,
        input_f,
        mxl_folder,
        midi_folder,
        name_suffix,
        use_constraints=False,
        intertwine_bar_info=False, # no bar default
        normalize_tonality=False,
        temperature=1.0,
        p=0.9,
        unmasking_order='random',
        num_stages=None,
        use_conditions=False,
        create_gen=True,
        create_real=False
    ):
    # we cannot have intertwine_bar_info == True and use_constraints == False
    # because bar information is passed through the constraints
    # if intertwine_bar_info:
    #     use_constraints = True

    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
            input_f,
            keep_durations=True,
            normalize_tonality=normalize_tonality,
        )

    harmony_real = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    harmony_input = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    # if intertwine_bar_info is True and use_constraints is False, we only need to pass
    # the bar information as a constraint, not the chords, or anything else
    # so mask out everything except from bar_token_ids
    if intertwine_bar_info and not use_constraints:
        harmony_input[ harmony_input != tokenizer.bar_token_id ] = tokenizer.mask_token_id
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    if use_conditions:
        conditioning_vec = torch.FloatTensor( input_encoded['time_signature'] ).reshape( 1, len(input_encoded['time_signature']) )
    else:
        conditioning_vec = None
    
    if create_gen:
        random_generated_harmony = nucleus_token_by_token_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            mask_token_id=tokenizer.mask_token_id,
            temperature=temperature,
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True,         # disallow <pad>/<nc> before melody ends
            chord_constraints = harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
            p=p,
            unmasking_order=unmasking_order,
            num_stages=num_stages,
            conditioning_vec=conditioning_vec
        )
        gen_output_tokens = []
        for t in random_generated_harmony[0].tolist():
            gen_output_tokens.append( tokenizer.ids_to_tokens[t] )
    else:
        avg_diffs = None
        gen_output_tokens = None
    # keep ground truth
    harmony_real_tokens = []
    for t in harmony_real[0].tolist():
        harmony_real_tokens.append( tokenizer.ids_to_tokens[t] )
    gen_score = None
    real_score = None
    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded['melody_part'],
            gen_output_tokens,
            input_encoded['ql_per_quantum'],
            input_encoded['skip_steps']
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded['back_interval'])
        if mxl_folder is not None:
            mxl_file_name = os.path.join(mxl_folder, f'gen_{name_suffix}' + '.mxl')
            save_harmonized_score(gen_score, out_path=mxl_file_name)
        if midi_folder is not None:
            midi_file_name = os.path.join(midi_folder, f'gen_{name_suffix}' + '.mid')
            save_harmonized_score(gen_score, out_path=midi_file_name)
        # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')
    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded['melody_part'],
            harmony_real_tokens,
            input_encoded['ql_per_quantum'],
            input_encoded['skip_steps']
        )
        
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded['back_interval'])
        if mxl_folder is not None:
            mxl_file_name = os.path.join(mxl_folder, f'real_{name_suffix}' + '.mxl')
            save_harmonized_score(real_score, out_path=mxl_file_name)
        if midi_folder is not None:
            midi_file_name = os.path.join(midi_folder, f'real_{name_suffix}' + '.mid')
            save_harmonized_score(real_score, out_path=midi_file_name)
        # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return gen_output_tokens, harmony_real_tokens, gen_score, real_score
# end generate_files_with_nucleus