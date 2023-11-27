import json
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_curve_classes(root: str, partition: str='train', parameter: str='accuracy', dataset: list=None):
    '''
        Draws data per step.
    '''
    dict_classes = {
        0: 'B-mount',
        1: 'I-mount',
        2: 'O'
    }
    figure, axes = plt.subplots(layout='constrained')

    index = 0
    for data in dataset:
        line, = axes.plot(np.arange(len(data)) + 1, data)
        line.set_label(dict_classes[index])

        index += 1
    
    axes.set_title(parameter)
    axes.legend()
    figure.savefig(os.path.join(root, partition + '_' + parameter + '.png'))


def draw_curve_train_val_pair(root: str, parameter: str='loss', data_train: list=None, data_val: list=None):
    '''
        Draws data per epoch.
    '''
    x = np.arange(len(data_train))
    figure, axes = plt.subplots()
    axes.plot(x, data_train, label='train')
    axes.plot(x, data_val, label='val')

    axes.set_title(parameter)
    axes.legend()
    figure.savefig(os.path.join(root, parameter + '.png'))


def visualize_data_pr(log_file_path: str, output_folder: str, partition: str='train'):
    p_list_0, p_list_1, p_list_2 = [], [], []
    r_list_0, r_list_1, r_list_2 = [], [], []

    with open(log_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            p_list_0.append(data['precision']['0'])
            p_list_1.append(data['precision']['1'])
            p_list_2.append(data['precision']['2'])
            r_list_0.append(data['recall']['0'])
            r_list_1.append(data['recall']['1'])
            r_list_2.append(data['recall']['2'])

    draw_curve_classes(output_folder, partition, 'precision', [p_list_0, p_list_1, p_list_2])
    draw_curve_classes(output_folder, partition, 'recall', [r_list_0, r_list_1, r_list_2])


def visualize_data_la(train_log_path: str, val_log_path: str, output_folder: str):
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []

    with open(train_log_path, 'r') as f:
        temp_loss, temp_accuracy = [], []
        prev_epoch = 0
        for line in f:
            data = json.loads(line)

            if data['epoch'] == prev_epoch:
                temp_loss.append(data['loss'])
                temp_accuracy.append(data['accuracy'])
            else:
                prev_epoch = data['epoch']
                train_loss.append(np.mean(temp_loss))
                train_accuracy.append(np.mean(temp_accuracy))

                temp_accuracy.clear()
                temp_loss.clear()

                temp_loss.append(data['loss'])
                temp_accuracy.append(data['accuracy'])
        train_loss.append(np.mean(temp_loss))
        train_accuracy.append(np.mean(temp_accuracy))


    with open(val_log_path, 'r') as f:
        temp_loss, temp_accuracy = [], []
        prev_epoch = 0
        for line in f:
            data = json.loads(line)

            if data['epoch'] == prev_epoch:
                temp_loss.append(data['loss'])
                temp_accuracy.append(data['accuracy'])
            else:
                prev_epoch = data['epoch']
                val_loss.append(np.mean(temp_loss))
                val_accuracy.append(np.mean(temp_accuracy))

                temp_accuracy.clear()
                temp_loss.clear()

                temp_loss.append(data['loss'])
                temp_accuracy.append(data['accuracy'])

        val_loss.append(np.mean(temp_loss))
        val_accuracy.append(np.mean(temp_accuracy))

    draw_curve_train_val_pair(output_folder, parameter='loss', data_train=train_loss, data_val=val_loss)
    draw_curve_train_val_pair(output_folder, parameter='accuracy', data_train=train_accuracy, data_val=val_accuracy)


def visualize_logs(train_log_path: str, val_log_path: str, output_folder: str):
    visualize_data_pr(log_file_path=train_log_path, output_folder=output_folder, partition='train')
    visualize_data_pr(log_file_path=val_log_path, output_folder=output_folder, partition='val')

    visualize_data_la(train_log_path=train_log_path, val_log_path=val_log_path, output_folder=output_folder)