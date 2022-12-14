import torch
import torch.nn as nn
import sys
import os
import argparse
from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.training.trainer import Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. rt')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=10, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.00001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=[3], nargs='+', help="Specify scheduler cycle, e.g. 10 100 1000")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--aug', action='store_true', help='use data augmentation')
    commandLineParser.add_argument('--aug_num', type=int, default=3, help="Number of times to augment")
    commandLineParser.add_argument('--aug_method', type=str, default='eda', help="Augmentation Method")
    commandLineParser.add_argument('--change_amount', type=float, default=0.5, help="Fraction of words to change in augmentation for example")
    commandLineParser.add_argument('--prune', type=float, default=1.0, help="Pruning: Fraction of samples to keep")
    commandLineParser.add_argument('--not_pretrained', action='store_true', help='do not use pretrained_model')
    commandLineParser.add_argument('--aug_sample', action='store_true', help='use data augmentation to define a distribution and use this to sample original training samples')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}_aug{args.aug}_aug-sample{args.aug_sample}_aug-num{args.aug_num}_change-amount{args.change_amount}_method-{args.aug_method}_pretrained{not args.not_pretrained}_prune{args.prune}_seed{args.seed}.th'

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data - change this part of code for the aug sample experiments
    val_data, train_data = select_data(args, train=True, aug=args.aug, aug_num=args.aug_num, change_amount=args.change_amount, aug_method=args.aug_method)

    # Initialise model
    model = select_model(args.model_name, pretrained=not args.not_pretrained)
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    trainer = Trainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(train_data, val_data, out_file, max_epochs=args.epochs, bs=args.bs)