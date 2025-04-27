from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
import numpy as np
from torch.utils.data import DataLoader
from models import GridMLMMelHarm
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse
from train_utils import train_with_curriculum

curriculum_types = ['no', 'random', 'ts_blank', 'ts_incr']

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a GridMLM model with a specific curriculum type.')

    # Define arguments
    parser.add_argument('-c', '--curriculum', type=str, help='Specify the curriculum type name among: ' + repr(curriculum_types), required=True)
    parser.add_argument('-s', '--stages', type=int, help='Specify the number of epochs per stage in the curriculum.', required=True)
    parser.add_argument('-p', '--progression', type=str, help='Specify curriculum progression: "linear", "mixed", "uniform".', required=True)
    parser.add_argument('-a', '--stage_aware', type=int, help='Model is stage aware (1) or not (0).', required=True)
    parser.add_argument('-d', '--datatrain', type=str, help='Specify the full path to the root folder of the training xml/mxl files', required=True)
    parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 5e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 8.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    curriculum_type = args.curriculum
    curriculum_progression = args.progression
    epochs_per_stage = args.stages
    stage_aware = args.stage_aware != 0
    # root_dir = '/media/maindisk/maximos/data/hooktheory_xmls'
    train_dir = args.datatrain
    val_dir = args.dataval
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 50
    if args.epochs:
        epochs = args.epochs
    lr = 5e-5
    if args.learningrate:
        lr = args.learningrate
    batchsize = 8
    if args.batchsize:
        batchsize = args.batchsize

    tokenizer = CSGridMLMTokenizer(fixed_length=256)

    train_dataset = CSGridMLMDataset(train_dir, tokenizer, 512)
    val_dataset = CSGridMLMDataset(val_dir, tokenizer, 512)

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=CSGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=CSGridMLM_collate_fn)

    loss_fn=CrossEntropyLoss(ignore_index=-100)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    model = GridMLMMelHarm(
        chord_vocab_size=len(tokenizer.vocab),
        conditioning_dim=16,
        device=device,
        max_stages=6
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/' + curriculum_progression + '/', exist_ok=True)
    results_path = 'results/' + curriculum_progression + '/' + 'a_'*stage_aware + curriculum_type + '.csv'
    
    save_dir = 'saved_models/' + curriculum_progression + '/'
    os.makedirs(save_dir, exist_ok=True)
    transformer_path = save_dir + 'a_'*stage_aware + curriculum_type + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,  # 'no', 'random', 'ts_blank', 'ts_incr'
        curriculum_progression=curriculum_progression, # 'linear', 'mixed', 'uniform'
        epochs_per_stage=epochs_per_stage,
        results_path=results_path,
        transformer_path=transformer_path,
        stage_aware=stage_aware
    )
    
# end main

if __name__ == '__main__':
    main()