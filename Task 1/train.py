import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from bert import BertModel, BertDataset, BertTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', default=15, type=int, 
                        help='Flag indicates number of epochs in training loop.')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='Sets batch-size.')
    parser.add_argument('--dataset', type=str,
                        help='Path to the dataset.')
    

    return parser.parse_args()


def find_positive(labels, logits):
    total_positive = 0
    number_tokens = 0
    for i in range(logits.shape[0]):

        logits_clean = logits[i][labels[i] != -100]
        label_clean = labels[i][labels[i] != -100]

        predictions = logits_clean.argmax(dim=1)
        number_tokens += len(predictions)
        total_positive += torch.sum((predictions == label_clean))

    return total_positive, number_tokens


def train_step(model, optimizer, input_ids, attention_masks, labels):
    model.train()
    
    optimizer.zero_grad()
    loss, logits = model(input_ids, attention_masks, labels)

    loss.backward()
    optimizer.step()

    return loss, logits


def val_step(model, input_ids, attention_masks, labels):
    model.eval()

    loss, logits = model(input_ids, attention_masks, labels)

    return loss, logits


def training_loop(model, optimizer, epochs:int, data_train: DataLoader, data_val: DataLoader, device: str='cpu'):
    print('Start training...')
    for epoch in range(epochs):
        equal_tokens = 0
        total_tokens = 0
        losses = []

        # Training step
        for input_ids, attention_masks, labels in data_train:
            input_ids = input_ids.squeeze(axis=1).to(device)
            attention_masks = attention_masks.squeeze(axis=1).to(device)
            labels = labels.to(device)

            loss, logits = train_step(model, optimizer, input_ids, attention_masks, labels)
            positive_tokens, tokens = find_positive(labels, logits)
            
            equal_tokens += positive_tokens
            total_tokens += tokens
            
            losses.append(loss.item())

            print('Training step. Accuracy on the Epoch loss is: ', epoch, equal_tokens / total_tokens, np.mean(losses), end='\r')
        print()


        equal_tokens = 0
        total_tokens = 0
        losses = []
        with torch.no_grad():
            # Validation step
            for input_ids, attention_masks, labels in data_val:
                input_ids = input_ids.squeeze(axis=1).to(device)
                attention_masks = attention_masks.squeeze(axis=1).to(device)
                labels = labels.to(device)

                loss, logits = val_step(model, input_ids, attention_masks, labels)

                positive_tokens, tokens = find_positive(labels, logits)
                
                equal_tokens += positive_tokens
                total_tokens += tokens
                
                losses.append(loss.item())
                print('Validation step. Accuracy is %f, loss is: %f                                       ' 
                    % (equal_tokens / total_tokens, np.mean(losses)), end='\r')
            print()


if __name__ == '__main__':
    args = parse_arguments()
    print(args)