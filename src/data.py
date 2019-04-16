import random

import pandas as pd
import pickle
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms

from src.config import BASE_SIZE, BRANCH_NUM, CAPTIONS, END_TOKEN, CAP_MAX_LEN


class CUB(Dataset):
    def __init__(self):
        print('Loading image paths ...')
        self.data = pd.read_csv('CUB_200_2011/images.txt', delim_whitespace=True, header=None, index_col=0,
                                names=['img_path'])

        print('Loading image captions, counting word frequencies, building vocab ...')
        self.word_freq = {}
        self.captions = {}
        for i, row in self.data.iterrows():
            dir, file = row['img_path'].split('/')
            file = file.replace('.jpg', '.txt')

            pattern = re.compile(r'[^\sa-z]+')
            with open(f'CUB_200_2011/text/{dir}/{file}', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Keep only alphanumeric characters
                captions = [pattern.sub('', line.lower().strip()) for line in lines]
                self.captions[i] = captions

                for c in captions:
                    for word in c.split():
                        self.word_freq[word] = self.word_freq.get(word, 0) + 1

        self.vocab = {word_freq[0]: i for i, word_freq in enumerate(self.word_freq.items(), start=1)
                      # if word_freq[1] > 2
                      }
        self.vocab[END_TOKEN] = 0
        print(f'Vocab size:{len(self.vocab)}')

        print('Setting train/test split...')
        with open('CUB_200_2011/test/filenames.pickle', 'rb') as f:
            test = pickle.load(f)

        self.data['train'] = self.data['img_path'].apply(lambda x: 0 if x[:x.index('.jpg')] in test else 1)
        self.data.sort_values(by='train', inplace=True, ascending=False)

        print('Loading bounding boxes ...')
        bbox = pd.read_csv('CUB_200_2011/bounding_boxes.txt', delim_whitespace=True, header=None, index_col=0,
                           names=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])
        self.data = self.data.merge(bbox, left_index=True, right_index=True)

        self.max_size = BASE_SIZE * (2 ** (BRANCH_NUM - 1))
        self.transforms = transforms.Compose([
            transforms.Resize(self.max_size * 76 // 64),
            transforms.RandomCrop(self.max_size),
            transforms.RandomHorizontalFlip()
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imsize = [BASE_SIZE * 2 ** i for i in range(BRANCH_NUM)]

    def get_caption(self, index):
        caption_idx = random.randint(0, CAPTIONS - 1)
        caption = self.captions[index][caption_idx]
        encoded = [self.vocab[w] for w in caption.split()]
        cap_len = len(encoded)
        if cap_len < CAP_MAX_LEN:
            encoded = encoded + [self.vocab[END_TOKEN] for _ in range(CAP_MAX_LEN - cap_len)]

        return encoded[:CAP_MAX_LEN], cap_len

    def get_image(self, index):
        img_data = self.data.loc[index, :]
        img = Image.open('CUB_200_2011/images/' + img_data.img_path)
        width, height = img.size

        r = int(np.maximum(img_data.bbox_width, img_data.bbox_height) * 0.75)
        # center_x = (2 * img_data.bbox_x + img_data.bbox_width) // 2
        center_x = img_data.bbox_x + img_data.bbox_width // 2
        center_y = img_data.bbox_y + img_data.bbox_height // 2
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        img = self.transforms(img)

        imgs = []
        for i in range(BRANCH_NUM):
            if i < (BRANCH_NUM - 1):
                re_img = transforms.Resize(self.imsize[i])(img)
            else:
                re_img = img
            imgs.append(self.normalize(re_img))

        return imgs

    def __getitem__(self, index):
        index = index + 1  # Image index starts from 1
        imgs = self.get_image(index)
        caption, cap_len = self.get_caption(index)
        return imgs, caption, cap_len

    def __len__(self):
        return self.data.train.sum()
