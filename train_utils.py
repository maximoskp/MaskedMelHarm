import torch
import random
from tqdm import tqdm
from data_utils import compute_normalized_token_entropy
# TODO:
# validation loop
# save model
# write results to excel
# integrate compute_normalized_token_entropy and perplexity

def apply_structured_masking(harmony_tokens, mask_token_id, step_idx, time_sigs, incremental=False):
    """
    harmony_tokens: (B, 256) - original ground truth tokens
    mask_token_id: int - ID for the special <mask> token
    step_idx: int - 0, 1, 2, etc.
    time_sigs: batch of 16-item binary encodings of time signature for each item

    Returns:
        masked_harmony: (B, 256) with some tokens replaced with <mask>
        target_harmony: (B, 256) with -100 at positions we do NOT want loss
    """
    B, T = harmony_tokens.shape

    # Create masked version
    masked_harmony = torch.full_like( harmony_tokens, mask_token_id )

    # Create target (loss computed only on masked positions)
    target = harmony_tokens.clone()
    device = harmony_tokens.device
    mask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )
    # mask = torch.zeros((T,), dtype=torch.bool, device=device)  # start with all masked
    # Expand to batch
    # mask = mask.unsqueeze(0).expand(B, -1)  # shape (B, 256)
    if incremental and step_idx > 0:
        input_mask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )

    for i in range(B):
        # get ts num and den
        ts_num = torch.nonzero(time_sigs[i, :14])[0] + 2
        ts_den = torch.nonzero(time_sigs[i, 14:])[0]*8 - 4
        spacing_schedule = [ 4*ts_num*(ts_den//4) , 2*ts_num*(ts_den//4) , 1*ts_num*(ts_den//4) ]
        spacing = spacing_schedule[step_idx] if step_idx < len(spacing_schedule) else 1

        # Get the indices that will remain unmasked for this step
        mask[i, ::spacing] = True  # reveal tokens at spacing in target
        if incremental and step_idx > 0:
            spacing = spacing_schedule[step_idx-1] if step_idx-1 < len(spacing_schedule) else spacing_schedule[-1]
            input_mask[i, ::spacing] = True  # reveal tokens at spacing in harmony input
    target[~mask] = -100  # ignore tokens that were shown to the model
    if incremental and step_idx > 0:
        masked_harmony[input_mask] = harmony_tokens[input_mask]
    return masked_harmony, target
# end apply_structured_masking

def get_step_idx_linear(epoch, epochs_per_stage, max_step_idx):
    return min(epoch // epochs_per_stage, max_step_idx)
# end get_step_idx_linear

def get_step_idx_mixed(epoch, max_epoch, max_step_idx):
    """Returns a random step index, biased toward early stages in early epochs."""
    progress = epoch / max_epoch
    probs = torch.softmax(torch.tensor([
        (1.0 - abs(progress - (i / max_step_idx))) * 5 for i in range(max_step_idx + 1)
    ]), dim=0)
    return torch.multinomial(probs, 1).item()
# end get_step_idx_mixed

def train_with_curriculum(
    model, optimizer, trainloader, loss_fn, mask_token_id,
    epochs=10,
    curriculum_type="linear",  # "linear" or "mixed"
    epochs_per_stage=2,
):
    device = next(model.parameters()).device
    max_step_idx = 2

    for epoch in range(epochs):
        model.train()

        # Determine masking level
        if curriculum_type == "linear":
            step_idx = get_step_idx_linear(epoch, epochs_per_stage, max_step_idx)
        elif curriculum_type == "mixed":
            step_idx = get_step_idx_mixed(epoch, epochs, max_step_idx)
        else:
            raise ValueError("Invalid curriculum type")

        epoch_loss = 0.0
        running_loss = 0.
        running_accuracy = 0.
        batch_num = 0
        with tqdm(trainloader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} (step {step_idx}) | trn')
            for batch in tepoch:
                melody_grid = batch["pianoroll"].to(device)           # (B, 256, 140)
                harmony_gt = batch["input_ids"].to(device)         # (B, 256)
                conditioning_vec = batch["time_signature"].to(device)  # (B, C0)
                
                # Apply masking to harmony
                harmony_input, harmony_target = apply_structured_masking(
                    harmony_gt,
                    mask_token_id,
                    step_idx,
                    conditioning_vec
                )

                # Forward pass
                logits = model(conditioning_vec, melody_grid, harmony_input)

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                train_accuracy = running_accuracy/batch_num

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
            # end for batch
        # end with tqdm
    # emd for epoch
# end train_with_curriculum