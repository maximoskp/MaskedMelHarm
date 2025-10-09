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

curriculum_types = ['random', 'base2']

def train_qt_bar_gmlm(
        train_dataset,
        trainloader,
        valloader,
        tokenizer,
        curriculum_type,
        total_stages,
        subfolder,
        device_name,
        epochs=100,
        lr=1e-5,
        batchsize=8,
        validations_per_epoch=1,
        tqdm_position=0
    ):

# def main():

#     # Create the argument parser
#     parser = argparse.ArgumentParser(description='Script for training a GridMLM model with a specific curriculum type.')

#     # Define arguments
#     parser.add_argument('-c', '--curriculum', type=str, help='Specify the curriculum type name among: ' + repr(curriculum_types), required=True)
#     parser.add_argument('-s', '--total_stages', type=int, help='Specify number of stages, applicable to random only.', required=False)
#     parser.add_argument('-f', '--subfolder', type=str, help='Specify subfolder to save the model and results.', required=False)
#     parser.add_argument('-d', '--datatrain', type=str, help='Specify the full path to the root folder of the training xml/mxl files', required=True)
#     parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
#     parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
#     parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
#     parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 5e-5.', required=False)
#     parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 8.', required=False)
    
#     # Parse the arguments
#     args = parser.parse_args()
#     curriculum_type = args.curriculum
#     total_stages = 10
#     if args.total_stages and curriculum_type == 'random':
#         total_stages = args.total_stages
#     elif args.total_stages and curriculum_type == 'step':
#         total_stages = args.total_stages
#     subfolder = ''
#     if args.subfolder:
#         subfolder = args.subfolder
#     train_dir = args.datatrain
#     val_dir = args.dataval
#     device_name = 'cpu'
#     if args.gpu is not None:
#         if args.gpu > -1:
#             device_name = 'cuda:' + str(args.gpu)
#     epochs = 50
#     if args.epochs:
#         epochs = args.epochs
#     lr = 5e-5
#     if args.learningrate:
#         lr = args.learningrate
#     batchsize = 8
#     if args.batchsize:
#         batchsize = args.batchsize

    # tokenizer = CSGridMLMTokenizer(fixed_length=80, quantization='4th', intertwine_bar_info=True, trim_start=False)

    # train_dataset = CSGridMLMDataset(train_dir, tokenizer, name_suffix='MLMH_bar_qt')
    # val_dataset = CSGridMLMDataset(val_dir, tokenizer, name_suffix='MLMH_bar_qt')

    # trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=CSGridMLM_collate_fn)
    # valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=CSGridMLM_collate_fn)

    def compute_class_weights_from_dataset(dataset, tokenizer, scheme="temp", alpha=0.5, beta=0.999, ignore_index=-100):
        """
        Compute class weights for CrossEntropyLoss given a dataset of chord tokens.
        
        Args:
            dataset: dataset loaded from pickle
            tokenizer: tokenizer object with vocab (for num_classes)
            scheme: one of ["inv", "inv_sqrt", "cb", "temp"]
            alpha: used in temperature-scaled scheme
            beta: used in class-balanced scheme
            ignore_index: value used to skip padding/masked tokens
        
        Returns:
            torch.Tensor of shape [num_classes] with normalized class weights
        """
        num_classes = len(tokenizer.vocab)
        counts = torch.zeros(num_classes, dtype=torch.float)

        # Count occurrences of chord tokens across dataset
        for item in dataset:
            tokens = torch.tensor(item["input_ids"], dtype=torch.long)
            tokens = tokens[tokens != ignore_index]  # filter padding/masked tokens
            counts += torch.bincount(tokens, minlength=num_classes).float()

        freqs = counts / counts.sum()

        # Apply weighting scheme
        if scheme == "inv":  # 1/p
            weights = 1.0 / (freqs + 1e-9)

        elif scheme == "inv_sqrt":  # 1/sqrt(p)
            weights = 1.0 / torch.sqrt(freqs + 1e-9)

        elif scheme == "cb":  # Class-balanced loss
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-9)

        elif scheme == "temp":  # temperature-scaled: p^-alpha
            weights = (freqs + 1e-9) ** (-alpha)

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Normalize so average weight = 1
        weights = weights / weights.mean()

        return weights
    # end compute_class_weights_from_dataset

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    # loss_fn=CrossEntropyLoss(ignore_index=-100)
    # Precompute once before training
    class_weights = compute_class_weights_from_dataset(
        train_dataset, tokenizer, scheme="temp", alpha=0.5
    )

    # Define loss function with weights
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device), ignore_index=-100
    )

    model = GridMLMMelHarm(
        d_model=512, 
        nhead=8, 
        num_layers=8, 
        chord_vocab_size=len(tokenizer.vocab),
        device=device,
        grid_length=80,
        max_stages=total_stages,
        conditioning_dim=8 + (curriculum_type == 'step'),
        pianoroll_dim=tokenizer.pianoroll_dim,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/bar_qt/', exist_ok=True)
    os.makedirs('results/bar_qt/' + subfolder + '/', exist_ok=True)
    if curriculum_type == 'random':
        results_path = 'results/bar_qt/' + subfolder + '/' + curriculum_type + str(total_stages) + '.csv'
    else:
        results_path = 'results/bar_qt/' + subfolder + '/' + curriculum_type + '.csv'

    os.makedirs('saved_models/bar_qt/', exist_ok=True)
    os.makedirs('saved_models/bar_qt/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/bar_qt/' + subfolder + '/'
    if curriculum_type == 'random':
        transformer_path = save_dir + curriculum_type + str(total_stages) + '.pt'
    else:
        transformer_path = save_dir + curriculum_type + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,  # 'random', 'base2'
        total_stages=total_stages,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id,
        condition='h_density_complexity',
        validations_per_epoch=validations_per_epoch,
        tqdm_position=tqdm_position
    )
    
# end main

# if __name__ == '__main__':
#     main()