import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from bert import BertModel, BertDataset, BertTokenizer
from ast import literal_eval


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', default=15, type=int, 
                        help='Flag indicates number of epochs in training loop.')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='Sets batch-size.')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Option for specifying learning rate in SGD.')
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


def find_recall(labels, logits, expected_token: int) -> torch.float32:
    temp_recall = []

    for i in range(logits.shape[0]):
        logits_clean = logits[i][labels[i] != -100]
        label_clean = labels[i][labels[i] != -100]

        predictions = logits_clean.argmax(dim=1)

        positive = 0
        all = 0
        for index, token in enumerate(label_clean):
            if token == expected_token:
                all += 1

                if token == predictions[index]:
                    positive += 1

        if all == 0:
            temp_recall.append(1)
        else:
            temp_recall.append(positive / all)

    return np.mean(temp_recall)


def find_precision(labels, logits, expected_token: str) -> torch.float32:
    temp_precision = []

    for i in range(logits.shape[0]):
        logits_clean = logits[i][labels[i] != -100]
        label_clean = labels[i][labels[i] != -100]

        predictions = logits_clean.argmax(dim=1)

        positive = 0
        negative = 0
        for index, token in enumerate(predictions):
            if token == expected_token:
                if token == label_clean[index]:
                    positive += 1
                else:
                    negative += 1
        
        if (positive + negative) == 0:
            temp_precision.append(1)
        else:
            temp_precision.append(positive / (negative + positive))

    return np.mean(temp_precision)


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
        recalls = {
            0: [],
            1: [],
            2: []
        }
        precisions = {
            0: [],
            1: [],
            2: []
        }

        # Training step
        for input_ids, attention_masks, labels in data_train:
            input_ids = input_ids.squeeze(axis=1).to(device)
            attention_masks = attention_masks.squeeze(axis=1).to(device)
            labels = labels.to(device)

            loss, logits = train_step(model, optimizer, input_ids, attention_masks, labels)
            positive_tokens, tokens = find_positive(labels, logits)

            for key, value in recalls.items():
                value.append(find_recall(logits=logits, labels=labels, expected_token=key))
            
            for key, value in precisions.items():
                value.append(find_precision(logits=logits, labels=labels, expected_token=key))
            
            equal_tokens += positive_tokens
            total_tokens += tokens
            
            losses.append(loss.item())

            print('Training step. Accuracy on the Epoch %d is %f loss is %f' % (epoch, equal_tokens / total_tokens, np.mean(losses)), '                         ', end='\r')
        print()
        print('\tRecall: ', {key: np.mean(value) for key, value in recalls.items()})
        print('\tPrecision: ', {key: np.mean(value) for key, value in recalls.items()}) 


        equal_tokens = 0
        total_tokens = 0
        losses = []
        recalls = {
            0: [],
            1: [],
            2: []
        }
        with torch.no_grad():
            # Validation step
            for input_ids, attention_masks, labels in data_val:
                input_ids = input_ids.squeeze(axis=1).to(device)
                attention_masks = attention_masks.squeeze(axis=1).to(device)
                labels = labels.to(device)

                loss, logits = val_step(model, input_ids, attention_masks, labels)

                positive_tokens, tokens = find_positive(labels, logits)
            for key, value in recalls.items():
                value.append(find_recall(logits=logits, labels=labels, expected_token=key))
                
                equal_tokens += positive_tokens
                total_tokens += tokens
                
                losses.append(loss.item())
                print('Validation step. Accuracy is %f, loss is: %f                                       ' 
                    % (equal_tokens / total_tokens, np.mean(losses)), end='\r')
            print()
            print('\tRecall: ', {key: np.mean(value) for key, value in recalls.items()})
            print('\tPrecision: ', {key: np.mean(value) for key, value in precisions.items()}) 
            print('\n')


if __name__ == '__main__':
    args = parse_arguments()

    model = BertModel(classes_num=3)
    
    data = pd.read_csv(args.dataset, converters={'tags': literal_eval})
    dataset_tags = ['B-mount', 'I-mount', 'O']
    tokenizer = BertTokenizer(tags_list=dataset_tags)

    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), [int(.92 * len(data)), int(.99 * len(data))])
    print('Number of elements in datasets: ', 
            {
                'df_train' : len(df_train),
                'df_val': len(df_val),
                'df_test': len(df_test)
            }
    )

    dataset_train = BertDataset(dataframe=df_train, tokenizer=tokenizer)
    dataset_val = BertDataset(dataframe=df_val, tokenizer=tokenizer)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    training_loop(model, optimizer, args.epochs, dataloader_train, dataloader_val, device=device)

    model.bert.save_pretrained("./weights/bert_v2-test-4", from_pt=True)