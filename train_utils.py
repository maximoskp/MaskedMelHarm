import torch
import random
from tqdm import tqdm
from data_utils import compute_normalized_token_entropy
import random
# TODO:
# validation loop
# save model
# write results to excel
# integrate compute_normalized_token_entropy and perplexity

def apply_structured_masking(harmony_tokens,
    mask_token_id,
    step_idx,
    time_sigs,
    curriculum_type='no'):
    """
    harmony_tokens: (B, time_step) - original ground truth tokens
    mask_token_id: int - ID for the special <mask> token
    step_idx: int - 0, 1, 2, etc.
    time_sigs: batch of 16-item binary encodings of time signature for each item
    curriculum_type: how to progress with masking
    - 'no': all tokens are masked and the model needs to learn to unmask all
    - 'random': an increasing number of tokens is masked and the model needs to learn to unmask all
    - 'ts_blank': an increasing number of ts-based tokens is masked
    - 'ts_incr': an increasing number of ts-based tokens is masked while previous are unmasked

    Returns:
        masked_harmony: (B, time_steps) with some tokens replaced with <mask>
        target_harmony: (B, time_steps) with -100 at positions we do NOT want loss
    """
    B, T = harmony_tokens.shape

    # Create masked version that will serve as the input
    masked_harmony = torch.full_like( harmony_tokens, mask_token_id )

    # Create target. Loss computed only on masked positions that we care about learning
    # In incremental learning, we care to learn only a portion of the masked input tokens,
    # while for other masked tokens we don't care.
    target = harmony_tokens.clone()
    device = harmony_tokens.device
    # assume that not tokens need to be masked for learning
    target_to_learn = torch.full_like(
        harmony_tokens,
        curriculum_type=='random' or curriculum_type=='no',
        dtype=torch.bool,
        device=device
    )
    if curriculum_type == 'ts_incr' and step_idx > 0:
        # some tokens need to be revealed in the input for incremental learning
        input_unmask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )

    if curriculum_type == 'ts_incr' or curriculum_type == 'ts_blank':
        for i in range(B):
            # get ts num and den
            ts_num = torch.nonzero(time_sigs[i, :14])[0] + 2
            ts_den = torch.nonzero(time_sigs[i, 14:])[0]*8 - 4
            spacing_schedule = [ 4*ts_num*(ts_den//4) , 2*ts_num*(ts_den//4) , 1*ts_num*(ts_den//4) ]
            spacing = spacing_schedule[step_idx] if step_idx < len(spacing_schedule) else 1

            # Get the indices that will remain unmasked for this step
            target_to_learn[i, ::spacing] = True  # reveal tokens at spacing in target
            if curriculum_type == 'ts_incr' and step_idx > 0:
                spacing = spacing_schedule[step_idx-1] if step_idx-1 < len(spacing_schedule) else spacing_schedule[-1]
                input_unmask[i, ::spacing] = True  # reveal tokens at spacing in harmony input
    if curriculum_type == 'random':
        step_ratios = {0: 0.6, 1: 0.3, 2: 0}
        for i in range(B):
            if step_idx < 2:
                step_ratio = step_ratios[step_idx]
                valid_indices = (harmony_tokens[i] != -1).nonzero(as_tuple=False).squeeze()
                n_reveal = max(1, int(len(valid_indices) * step_ratio))
                reveal_indices = random.sample(valid_indices.tolist(), n_reveal)
                masked_harmony[i, reveal_indices] = harmony_tokens[i, reveal_indices]
            target_to_learn = masked_harmony == mask_token_id
    if curriculum_type == 'ts_incr' and step_idx > 0:
        masked_harmony[input_unmask] = harmony_tokens[input_unmask]
        target_to_learn[input_unmask] = False
    target[~target_to_learn] = -100  # ignore tokens that were shown to the model
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
    curriculum_type='no',  # 'no', 'random', 'ts-based'
    curriculum_steps='linear', # 'linear', mixed
    epochs_per_stage=2,
):
    device = next(model.parameters()).device
    max_step_idx = 2

    for epoch in range(epochs):
        model.train()

        # Determine masking level
        if curriculum_steps == "linear":
            step_idx = get_step_idx_linear(epoch, epochs_per_stage, max_step_idx)
        elif curriculum_steps == "mixed":
            step_idx = get_step_idx_mixed(epoch, epochs, max_step_idx)
        else:
            raise ValueError("Invalid curriculum steps")

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
                    conditioning_vec,
                    curriculum_type
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