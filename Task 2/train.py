from src.models import MagicPoint, MagicPointDataset
from src.trainer import MagicPointTrainer
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epoches', type=int, default=10,
                        help='The number of epochs to train the model.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Specifies which optimizer must be used during training.')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='The batch size to divide the dataset.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.004,
                        help='Learning rate in the SGD optimizer.')
    parser.add_argument('-m', '--model', type=str, default='MagicPoint',
                        help='Which model to train.')
    parser.add_argument('--dataset-path', type=str, default='magic_point_dataset.csv',
                        help='Path to the magic point CSV dataset.')
    
    return parser.parse_args()


def train_magic_point(args):
    model = MagicPoint()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        raise ValueError('Problems with optimizer! No such optimizer available at the moment!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = MagicPointTrainer(model=model, optimizer=optimizer, device=device)

    dataset = pd.read_csv(args.dataset_path)
    ds_train = MagicPointDataset(dataset[dataset['partition'] == 'training'])
    ds_val = MagicPointDataset(dataset[dataset['partition'] == 'validation'])
    ds_test = MagicPointDataset(dataset[dataset['partition'] == 'test'])

    trainer.train(epoches=args.epoches,
                  val_dataloader=DataLoader(dataset=ds_val, batch_size=1),
                  train_dataloader=DataLoader(dataset=ds_train, batch_size=args.batch_size, shuffle=True),
                  test_dataloader=DataLoader(dataset=ds_test, batch_size=1)
    )
    print('Training finished! Saving model...')

    torch.save(trainer.model.state_dict(), 'weights/magic_point-%d-%.5f.pth' % (args.epoches, args.learning_rate))



if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    if args.model == 'MagicPoint':
        train_magic_point(args=args)
    else:
        raise ValueError('Currently no such model for training supported!')