import torch
import numpy as np


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