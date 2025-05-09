import torch
import torch.nn.functional as F
from train_utils import apply_structured_masking

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