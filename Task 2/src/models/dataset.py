from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2


class MagicPointDataset(Dataset):
    '''
        Dataset accepts path to the folder that contains all images, and corresponding points.
    '''
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.data = dataframe.sample(frac=1)


    def __len__(self):
        return len(self.data)


    def _unwrap_raw_label(self, width, height, raw):
        label = np.zeros(shape=(height, width))
        for point in raw:
            label[int(point[1])][int(point[0])] = 1

        return label.astype(np.float32)


    def __getitem__(self, index):
        image_path = self.data.iloc[index]['image_path']
        label_path = self.data.iloc[index]['label_path']

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        raw_label = np.load(label_path)
        label = self._unwrap_raw_label(width=image.shape[1], height=image.shape[0], raw=raw_label)
        image = np.asarray([image]).astype(np.float32) # from HWC to CHW

        return image, label
