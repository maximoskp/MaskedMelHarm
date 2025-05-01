import torch
import torch.nn.functional as F
from train_utils import apply_structured_masking

import torch

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
            selected_positions = masked_positions[topk_indices]
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


def ts_iterative_unmasking(
        model,
        conditioning_vec,
        melody_grid,
        stage_aware,
        mask_token_id,
        num_stages=6
    ):
    device = model.device
    output_tokens = torch.full([1,256], 1).to(device)
    fake_gt_tokens = torch.full([1,256], 1).to(device)

    # Start with all masked tokens
    output_tokens[:] = mask_token_id

    for stage in range(num_stages):
        model.eval()
        # Apply masking to harmony
        _, harmony_target = apply_structured_masking(
            fake_gt_tokens,
            mask_token_id,
            stage,
            conditioning_vec,
            'ts_blank'
        )
        with torch.no_grad():
            logits = model(
                conditioning_vec.to(device),
                melody_grid.to(device),
                output_tokens.to(device),
                None if not stage_aware else stage
            )
            probs = F.softmax(logits, dim=-1)

        # Sample or argmax predictions
        sampled_tokens = torch.argmax(probs, dim=-1)

        output_tokens[ harmony_target != -100 ] = sampled_tokens[ harmony_target != -100 ]
    return output_tokens
# end mask_predict

def one_shot_generate(model, seq_length, mask_token_id):
    batch_size = 1  # or more if you want
    device = next(model.parameters()).device

    input_tokens = torch.full((batch_size, seq_length), mask_token_id, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tokens)
        output_tokens = torch.argmax(logits, dim=-1)

    return output_tokens
# end one_shot_generate