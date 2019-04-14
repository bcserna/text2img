import pandas as pd
import pickle
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms

from src.config import BASE_SIZE, BRANCH_NUM


class CUB(Dataset):
    def __init__(self):
        # Load image paths
        self.data = pd.read_csv('CUB_200_2011/images.txt', delim_whitespace=True, header=None, index_col=0,
                                names=['img_path'])

        # Load captions
        self.captions = {}
        for idx, row in self.data.iterrows():
            dir, file = row['img_path'].split('/')
            file = file.replace('.jpg', '.txt')

            pattern = re.compile(r'[^\sa-z]+')
            with open(f'CUB_200_2011/text/{dir}/{file}', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Keep only alphanumeric characters
                captions = [pattern.sub('', line.lower().strip()) for line in lines]
                self.captions[idx] = captions

        # Set train and test examples
        with open('CUB_200_2011/test/filenames.pickle', 'rb') as f:
            test = pickle.load(f)

        self.data['train'] = self.data['img_path'].apply(lambda x: 0 if x[:x.index('.jpg')] in test else 1)

        # Load bounding boxes
        bbox = pd.read_csv('CUB_200_2011/bounding_boxes.txt', delim_whitespace=True, header=None, index_col=0,
                           names=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])
        self.data = self.data.merge(bbox, left_index=True, right_index=True)

        self.max_size = BASE_SIZE * (2 ** (BRANCH_NUM - 1))
        self.transforms = transforms.Compose([
            transforms.Scale(self.max_size * 76 // 64),
            transforms.RandomCrop(self.max_size),
            transforms.RandomHorizontalFlip()
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imsize = [BASE_SIZE * 2 ** i for i in range(BRANCH_NUM)]

    def __getitem__(self, index):
        index = index + 1  # image index starts from 1

        img_data = self.data.iloc[index]
        img = Image.open(img_data.image_path)
        width, height = img.size

        r = int(np.maximum(img_data.bbox_width, img_data.bbox_height) * 0.75)
        center_x = (2 * img_data.bbox_x + img_data.bbox_width) // 2
        center_y = (2 * img_data.bbox_y + img_data.bbox_height) // 2
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        img = self.transforms(img)

        ret = []
        for i in range(BRANCH_NUM):
            if i < (BRANCH_NUM - 1):
                re_img = transforms.Scale(self.imsize[i])(img)
            else:
                re_img = img
            ret.append(self.normalize(re_img))

        return ret

    def __len__(self):
        return self.data.shape[0]
