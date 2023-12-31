import torch
import numpy as np


class MagicPointTrainer:
    def __init__(self, model, optimizer, threshold: float=0.5, device: str='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold


    def _val_step(self, images, labels):
        self.model.eval()

        logits, scores = self.model(images)
        loss = self._weighted_loss(logits.reshape(-1), labels.reshape(-1))

        return loss, logits, scores
    

    def _weighted_loss(self, logits, labels):
        one_indeces = (labels == 1).nonzero(as_tuple=True)[0]
        zero_indeces = (labels == 0).nonzero(as_tuple=True)[0]

        weight = torch.zeros(len(logits)).to(self.device)
        weight[one_indeces] = 1
        weight[zero_indeces] = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()

        result = loss_fn(logits, labels)
        return result


    def _train_step(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad()

        logits, scores = self.model(images)
        loss = self._weighted_loss(logits.reshape(-1), labels.reshape(-1))
        loss.backward()
        self.optimizer.step()

        return loss, logits, scores
    

    def _calculate_recall_experiment(self, scores, targets, epsilon=3):
        predictions = torch.round(scores)
        true_positives = 0
        false_negatives = 0

        for prediction, target in zip(predictions, targets):
            for keypoint_prediction, keypoint_target in zip(prediction, target):
                distance = torch.linalg.norm(keypoint_prediction - keypoint_target)

                if distance <= epsilon:
                    true_positives += 1
                else:
                    false_negatives += 1

        if true_positives + false_negatives == 0:
            return 0

        recall = true_positives / (true_positives + false_negatives)
        return recall
    

    def _calculate_accuracy(self, scores, labels):
        positive = torch.sum(torch.round(scores) == labels)
        all = torch.prod(torch.as_tensor(scores.shape))
        
        return (positive / all).item()
    

    def _calculate_recall(self, logits, labels):
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        positive = torch.sum(torch.logical_and(torch.round(logits) == 1, labels == 1))
        all = torch.sum(labels == 1)

        if all > 0:
            return (positive / all).item()
        return 1


    def _calculate_precision(self, logits, labels):
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        positive = torch.sum(torch.logical_and(labels == 1, torch.round(logits) == 1))
        all = torch.sum(torch.round(logits) == 1)

        if all > 0:
            return (positive / all).item()
        return 1


    def train(self, epoches: int, val_dataloader, train_dataloader, test_dataloader=None):
        print('Start training...')
        print()
        for epoch in range(epoches):
            epoch_dict = {
                'loss' : [],
                'accuracy' : [],
                'recall' : [],
                'precision': []
            }
            # Start training steps for the epoch
            for images, labels in train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                loss, logits, scores = self._train_step(images, labels)
                epoch_dict['loss'].append(loss.item())
                epoch_dict['accuracy'].append(self._calculate_accuracy(scores=scores, labels=labels))
                epoch_dict['precision'].append(self._calculate_precision(logits=scores, labels=labels))
                epoch_dict['recall'].append(self._calculate_recall(logits=scores, labels=labels))
                print('Training. For the epoch %d: ' % (epoch) + str({key: np.mean(value) for key, value in epoch_dict.items()}) + '            ', end='\r')
            print()

            for key in epoch_dict.keys():
                epoch_dict[key].clear()

            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    loss, logits, scores = self._val_step(images, labels)
                    epoch_dict['loss'].append(loss.item())
                    epoch_dict['accuracy'].append(self._calculate_accuracy(scores=scores, labels=labels))
                    epoch_dict['precision'].append(self._calculate_precision(logits=scores, labels=labels))
                    epoch_dict['recall'].append(self._calculate_recall(logits=scores, labels=labels))
                    print('Validating. For the epoch %d: '
                            % (epoch), {key: np.mean(value) for key, value in epoch_dict.items()}, 
                            '            ', end='\r')
                print()
                print()

        for key in epoch_dict.keys():
            epoch_dict[key].clear()
        # Testing the model
        if test_dataloader is not None:
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    loss, logits, scores = self._val_step(images, labels)
                    epoch_dict['loss'].append(loss.item())
                    epoch_dict['accuracy'].append(self._calculate_accuracy(scores=scores, labels=labels))
                    epoch_dict['precision'].append(self._calculate_precision(logits=scores, labels=labels))
                    epoch_dict['recall'].append(self._calculate_recall(logits=scores, labels=labels))
                    print('Testing. For the epoch %d: loss is %.4f, accuracy is %.4f'
                        % (epoch, np.mean(epoch_dict['loss']), np.mean(epoch_dict['accuracy'])))
                print()

        print('Training ended!')