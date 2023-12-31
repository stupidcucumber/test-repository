import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from bert import BertModel, BertDataset, BertTokenizer
from ast import literal_eval
import os
from datetime import datetime
import json
from bert import utils, visualize


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
    parser.add_argument('--device', type=str, default='cpu',
                        help='Specify the device on which you will train model.')
    parser.add_argument('--output_folder', type=str, default='bert-model',
                        help='Specify the name of a folder, which will contain final weights.')
    
    return parser.parse_args()


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


def log_state(file: str, **kwargs):
    file.write(json.dumps(kwargs))
    file.write('\n')
    file.flush()


def training_loop(model, optimizer, epochs:int, data_train: DataLoader, data_val: DataLoader, data_test: DataLoader, device: str='cpu'):
    LOG_FOLDER = 'data/logs/train'
    folder_path = os.path.join(LOG_FOLDER, datetime.now().strftime('train_%d_%m_%Y_%H_%M_%S'))
    os.mkdir(folder_path)
    log_file_train = open(os.path.join(folder_path, 'train.log'), 'a')
    log_file_val = open(os.path.join(folder_path, 'val.log'), 'a')
    log_file_test = open(os.path.join(folder_path, 'test.log'), 'a')


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
            positive_tokens, tokens = utils.find_positive(labels, logits)

            for key, value in recalls.items():
                value.append(utils.find_recall(logits=logits, labels=labels, expected_token=key))
            
            for key, value in precisions.items():
                value.append(utils.find_precision(logits=logits, labels=labels, expected_token=key))
            
            equal_tokens += positive_tokens
            total_tokens += tokens
            
            losses.append(loss.item())

            log_state(log_file_train, 
                  epoch=epoch, 
                  recall={key: np.mean(value) for key, value in recalls.items()}, 
                  precision={key: np.mean(value) for key, value in precisions.items()}, 
                  loss=np.mean(losses), 
                  accuracy=(equal_tokens / total_tokens).item())

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
        precisions = {
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

                positive_tokens, tokens = utils.find_positive(labels, logits)
                for key, value in recalls.items():
                    value.append(utils.find_recall(logits=logits, labels=labels, expected_token=key))

                for key, value in precisions.items():
                    value.append(utils.find_precision(logits=logits, labels=labels, expected_token=key))
                    
                equal_tokens += positive_tokens
                total_tokens += tokens
                    
                losses.append(loss.item())
                print('Validation step. Accuracy is %f, loss is: %f                                       ' 
                        % (equal_tokens / total_tokens, np.mean(losses)), end='\r')
                    
                log_state(log_file_val, 
                        epoch=epoch, 
                        recall={key: np.mean(value) for key, value in recalls.items()}, 
                        precision={key: np.mean(value) for key, value in precisions.items()}, 
                        loss=np.mean(losses), 
                        accuracy=(equal_tokens / total_tokens).item())
            print()
            print('\tRecall: ', {key: np.mean(value) for key, value in recalls.items()})
            print('\tPrecision: ', {key: np.mean(value) for key, value in precisions.items()}) 
            print('\n')
            

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
    # Testing our model
    print('Training has finished! Start testing...')
    with torch.no_grad():
        # Validation step
        for input_ids, attention_masks, labels in data_test:
            input_ids = input_ids.squeeze(axis=1).to(device)
            attention_masks = attention_masks.squeeze(axis=1).to(device)
            labels = labels.to(device)

            loss, logits = val_step(model, input_ids, attention_masks, labels)

            positive_tokens, tokens = utils.find_positive(labels, logits)
            for key, value in recalls.items():
                value.append(utils.find_recall(logits=logits, labels=labels, expected_token=key))
            for key, value in precisions.items():
                value.append(utils.find_precision(logits=logits, labels=labels, expected_token=key))
            
            equal_tokens += positive_tokens
            total_tokens += tokens
            
            losses.append(loss.item())
            print('Test step. Accuracy is %f, loss is: %f                                       ' 
                % (equal_tokens / total_tokens, np.mean(losses)), end='\r')
        print()
        print('\tRecall: ', {key: np.mean(value) for key, value in recalls.items()})
        print('\tPrecision: ', {key: np.mean(value) for key, value in precisions.items()}) 
        print('\n')
        log_state(log_file_test, 
                  epoch=epoch, 
                  recall=recalls, 
                  precision=precisions, 
                  loss=np.mean(losses),
                  accuracy=(equal_tokens / total_tokens).item())

    log_file_test.close()
    log_file_train.close()
    log_file_val.close()

    print('Fisualizing logs...')
    visualize.visualize_logs(train_log_path=os.path.join(folder_path, 'train.log'), 
                             val_log_path=os.path.join(folder_path, 'val.log'), 
                             output_folder=folder_path)


if __name__ == '__main__':
    args = parse_arguments()
    if os.path.exists(args.output_folder):
        raise Exception('Such folder already exists!')
    else:
        os.mkdir(args.output_folder)

    model = BertModel(classes_num=3)
    
    data = pd.read_csv(args.dataset, converters={'tags': literal_eval})
    dataset_tags = ['B-mount', 'I-mount', 'O']
    tokenizer = BertTokenizer(tags_list=dataset_tags)

    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), [int(.8 * len(data)), int(.9 * len(data))])
    print('Number of elements in datasets: ', 
            {
                'df_train' : len(df_train),
                'df_val': len(df_val),
                'df_test': len(df_test)
            }
    )

    dataset_train = BertDataset(dataframe=df_train, tokenizer=tokenizer)
    dataset_val = BertDataset(dataframe=df_val, tokenizer=tokenizer)
    dataset_test = BertDataset(dataframe=df_test, tokenizer=tokenizer)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    model = model.to(args.device)

    training_loop(model, optimizer, args.epochs, dataloader_train, dataloader_val, dataloader_test, device=args.device)

    model.bert.save_pretrained(args.output_folder, from_pt=True)
    print('Weights are saved!')