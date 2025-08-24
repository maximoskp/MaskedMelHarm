import torch
from torcheval.metrics.text import Perplexity
import random
from tqdm import tqdm
from data_utils import compute_normalized_token_entropy
import random
import csv
import numpy as np
import os
from transformers import get_cosine_schedule_with_warmup
# TODO:
# validation loop
# save model
# write results to excel
# integrate compute_normalized_token_entropy and perplexity

perplexity_metric = Perplexity(ignore_index=-100)

def random_progressive_masking(
        harmony_tokens,
        total_stages,
        mask_token_id,
        stage_in=None,
        bar_token_id=None
    ):
    """
    Generate visible input and denoising target for diffusion-style training.

    Args:
        harmony_tokens (torch.Tensor): Tensor of shape (B, L) containing target harmony token ids.
        stage (int): Current training stage (0 to total_stages - 1).
        total_stages (int): Total number of diffusion stages.
        mask_token_id (int): The token ID used to mask hidden positions in visible_harmony.
        device (str or torch.device): Target device.

    Returns:
        visible_harmony (torch.Tensor): Tensor of shape (B, L) with visible tokens (others masked).
        denoising_target (torch.Tensor): Tensor of shape (B, L) with tokens to predict (others = -100).
    """
    device = harmony_tokens.device
    B, L = harmony_tokens.shape

    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)  # -100 is ignored by CrossEntropyLoss

    if bar_token_id is not None:
        # Create a mask for bar token positions
        bar_mask = (harmony_tokens == bar_token_id)
        # Put bar tokens in visible_harmony (always unmasked)
        visible_harmony[bar_mask] = bar_token_id
        # Also include them in the denoising target (so model predicts them too)
        denoising_target[bar_mask] = bar_token_id
    
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    for b in range(B):
        stage = stage_indices[b] if stage_in is None else stage_in

        percent_visible = stage / total_stages
        percent_predict = 1 / total_stages
        perm = torch.randperm(L, device=device)
        num_visible = int(L * percent_visible)
        num_predict = int(L * percent_predict + 0.5)

        visible_idx = perm[:num_visible]
        predict_idx = perm[:num_visible + num_predict]  # predict includes visible + next unmasked tokens

        visible_harmony[b, visible_idx] = harmony_tokens[b, visible_idx]
        denoising_target[b, predict_idx] = harmony_tokens[b, predict_idx]

    return visible_harmony, denoising_target, stage_indices
# end random_progressive_masking

def structured_progressive_masking(
        harmony_tokens,
        total_stages,
        mask_token_id,
        stage_in=None
    ):
    B, L = harmony_tokens.shape
    device = harmony_tokens.device
    visible_harmony = torch.full_like( harmony_tokens, mask_token_id )
    denoising_target = harmony_tokens.clone()
    input_unmask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )
    target_to_learn = torch.full_like(
        harmony_tokens,
        False,
        dtype=torch.bool,
        device=device
    )
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    for i in range(B):
        stage = stage_indices[i] if stage_in is None else stage_in

        spacing_target = min( L, max(1, int((2**(8-stage)))) )
        # Get the indices that will remain unmasked for this step
        target_to_learn[i, ::spacing_target] = True  # reveal tokens at spacing in target
        spacing_input = 2*spacing_target #max(2, int((2**(8*(1-stage/total_stages)))*(ts_num/ts_den)))
        input_unmask[i, ::spacing_input] = spacing_input <= L  # reveal tokens at spacing in harmony input
    visible_harmony[input_unmask] = harmony_tokens[input_unmask]
    target_to_learn[input_unmask] = False
    denoising_target[~torch.logical_or( target_to_learn , input_unmask )] = -100  # ignore tokens that were not shown to the model
    return visible_harmony, denoising_target, stage_indices
# end structured_progressive_masking

def apply_masking(
        harmony_tokens,
        mask_token_id,
        total_stages=10,
        curriculum_type='random',
        stage=None,
        bar_token_id=None
    ):
    if curriculum_type == 'random':
        return random_progressive_masking(
                harmony_tokens,
                total_stages,
                mask_token_id,
                stage,
                bar_token_id
            )
    elif curriculum_type == 'base2':
        return structured_progressive_masking(
                harmony_tokens,
                total_stages,
                mask_token_id,
                stage
            )
# end apply_masking

def apply_structured_masking(harmony_tokens,
    mask_token_id,
    stage,
    time_sigs,
    total_stages=10,
    curriculum_type='no'):
    """
    harmony_tokens: (B, time_step) - original ground truth tokens
    mask_token_id: int - ID for the special <mask> token
    stage: int - stage of uncovering masks 0, 1, 2 ...
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
    if curriculum_type == 'ts_incr':
        # some tokens need to be revealed in the input for incremental learning
        input_unmask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )

    if curriculum_type == 'ts_incr' or curriculum_type == 'ts_blank':
        for i in range(B):
            # get ts num and den
            ts_num = torch.nonzero(time_sigs[i, :14])[0] + 2
            ts_den = torch.nonzero(time_sigs[i, 14:])[0]*8 - 4
            spacing_target = min(128, max(1, int((2**(7*stage/total_stages))*(ts_num/ts_den))))

            # Get the indices that will remain unmasked for this step
            target_to_learn[i, ::spacing_target] = True  # reveal tokens at spacing in target
            if curriculum_type == 'ts_incr':
                spacing_input = max(2, int((2**(8*stage/total_stages))*(ts_num/ts_den)))
                input_unmask[i, ::spacing_input] = True  # reveal tokens at spacing in harmony input
    if curriculum_type == 'random':
        for i in range(B):
            stage_ratio = 1. - (stage+1)/total_stages
            valid_indices = (harmony_tokens[i] != -1).nonzero(as_tuple=False).squeeze()
            n_reveal = int(len(valid_indices) * stage_ratio)
            reveal_indices = random.sample(valid_indices.tolist(), n_reveal)
            masked_harmony[i, reveal_indices] = harmony_tokens[i, reveal_indices]
            target_to_learn = masked_harmony == mask_token_id
    if curriculum_type == 'ts_incr':
        masked_harmony[input_unmask] = harmony_tokens[input_unmask]
        target_to_learn[input_unmask] = False
        target[~torch.logical_or( target_to_learn , input_unmask )] = -100  # ignore tokens that were not shown to the model
    if curriculum_type == 'ts_blank':
        target[~target_to_learn] = -100  # ignore tokens that were shown to the model
    return masked_harmony, target
# end apply_structured_masking

def get_stage_linear(epoch, epochs_per_stage, max_stage):
    return min(epoch // epochs_per_stage, max_stage)
# end get_stage_linear

def get_stage_mixed(epoch, max_epoch, max_stage):
    """Returns a random step index, biased toward early stages in early epochs.
    Last epoch is only with maximum stage."""
    if epoch >= max_epoch - 1:
        return max_stage
    progress = epoch / max_epoch
    probs = torch.softmax(torch.tensor([
        (1.0 - abs(progress - (i / max_stage))) * 5 for i in range(max_stage + 1)
    ]), dim=0)
    return torch.multinomial(probs, 1).item()
# end get_stage_mixed

def get_stage_uniform(epoch, max_epoch, max_stage):
    """Returns a random step index, uniform across all stages."""
    return np.random.randint(max_stage + 1)
# end get_stage_uniform

def validation_loop(model, valloader, mask_token_id, loss_fn, epoch, step, \
                    curriculum_type, train_loss, train_accuracy, \
                    train_perplexity, train_token_entropy,
                    best_val_loss, saving_version, results_path=None, transformer_path=None):
    device = model.device
    model.eval()
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        running_perplexity = 0
        val_perplexity = 0
        running_token_entropy = 0
        val_token_entropy = 0
        print('validation')
        with tqdm(valloader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch}/{step}| val')
            for batch in tepoch:
                perplexity_metric.reset()
                melody_grid = batch["pianoroll"].to(device)           # (B, 256, 140)
                harmony_gt = batch["input_ids"].to(device)         # (B, 256)
                conditioning_vec = batch["time_signature"].to(device)  # (B, C0)
                
                # Apply masking to harmony
                harmony_input, harmony_target, stage_indices = apply_masking(
                    harmony_gt,
                    mask_token_id,
                    total_stages=10,
                    curriculum_type=curriculum_type
                )

                # Forward pass
                logits = model(
                    conditioning_vec,
                    melody_grid,
                    harmony_input,
                    stage_indices
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                val_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                val_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                val_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                val_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
            # end for batch
    # end with tqdm
    if transformer_path is not None:
        if best_val_loss > val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_path)
    print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, train_loss, train_accuracy, \
                            train_perplexity, train_token_entropy, \
                            val_loss, val_accuracy, \
                            val_perplexity, val_token_entropy, \
                            saving_version] )
    return best_val_loss, saving_version
# end validation_loop

def train_with_curriculum(
    model, optimizer, trainloader, valloader, loss_fn, mask_token_id,
    epochs=100,
    curriculum_type='random',  # 'random', 'base2'
    total_stages=10,
    results_path=None,
    transformer_path=None,
    bar_token_id=None
):
    device = next(model.parameters()).device
    perplexity_metric.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', 'train_loss', 'train_acc', \
                        'train_ppl', 'train_te', 'val_loss', \
                        'val_acc', 'val_ppl', 'val_te', 'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        running_perplexity = 0
        train_perplexity = 0
        running_token_entropy = 0
        train_token_entropy = 0
        
        with tqdm(trainloader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch}/{step} | trn')
            for batch in tepoch:
                perplexity_metric.reset()
                model.train()
                melody_grid = batch["pianoroll"].to(device)           # (B, 256, 100)
                harmony_gt = batch["input_ids"].to(device)         # (B, 256)
                conditioning_vec = batch["time_signature"].to(device)  # (B, C0)
                
                # Apply masking to harmony
                harmony_input, harmony_target, stage_indices = apply_masking(
                    harmony_gt,
                    mask_token_id,
                    total_stages=total_stages,
                    curriculum_type=curriculum_type,
                    bar_token_id=None
                )

                # Forward pass
                logits = model(
                    conditioning_vec.to(device),
                    melody_grid.to(device),
                    harmony_input.to(device),
                    stage_indices
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                train_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                train_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                train_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
                step += 1
                if step%(total_steps//epochs) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_loop(
                        model,
                        valloader,
                        mask_token_id,
                        loss_fn,
                        epoch,
                        step,
                        curriculum_type,
                        train_loss,
                        train_accuracy,
                        train_perplexity,
                        train_token_entropy,
                        best_val_loss,
                        saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_with_curriculum