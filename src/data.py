import random
from collections import defaultdict
import pandas as pd
import pickle
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import glob

from src.config import BASE_SIZE, BRANCH_NUM, CAPTIONS, END_TOKEN, CAP_MAX_LEN, MIN_WORD_FREQ, UNK_TOKEN


def count_word_freq(df, freq=None):
    if freq is None:
        freq = defaultdict(int)
    for i, row in df.iterrows():
        for j in range(CAPTIONS):
            cap = row[f'caption_{j}']
            for w in cap.split():
                freq[w] += 1
    return freq


def build_vocab(word_freq, min_freq=MIN_WORD_FREQ, extra_tokens=[END_TOKEN, UNK_TOKEN]):
    vocab = {}
    for t in extra_tokens:
        vocab[t] = len(vocab)

    for w, f in word_freq.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def collate_fn(batch):
    ret = {
        'img64': [],
        'img128': [],
        'img256': [],
        'caption': [],
        'label': []
    }
    for img, cap, label in batch:
        ret['img64'].append(img[0])
        ret['img128'].append(img[1])
        ret['img256'].append(img[2])
        ret['caption'].append(cap)
        ret['label'].append(label)

    ret['img64'] = torch.stack(ret['img64'])
    ret['img128'] = torch.stack(ret['img128'])
    ret['img256'] = torch.stack(ret['img256'])
    return ret


class CUBSubset(Dataset):
    def __init__(self, subset, vocab, img_size, transformations, normalize, preload=False):
        self.data = subset
        self.vocab = vocab
        self.img_size = img_size
        self.transforms = transformations
        self.normalize = normalize
        self.images = None
        if preload:
            self.images = list(self.load_images())

    def load_images(self):
        for i in tqdm(range(len(self)), desc='Preloading images', dynamic_ncols=True):
            yield self.get_image(i)

    def __getitem__(self, index):
        imgs = self.get_image(index)
        caption = self.get_caption(index)
        label = self.data['class'].iloc[index]
        return imgs, caption, label

    def __len__(self):
        return self.data.shape[0]

    def get_caption(self, index):
        caption_idx = random.randint(0, CAPTIONS - 1)
        caption = self.data[f'caption_{caption_idx}'].iloc[index]
        return self.encode_text(caption)

    def encode_text(self, text):
        encoded = [self.vocab.get(w, self.vocab[UNK_TOKEN]) for w in text.split()]
        encoded = encoded[:CAP_MAX_LEN]  # address END token
        cap_len = len(encoded)
        encoded = encoded + [self.vocab[END_TOKEN]] * (CAP_MAX_LEN - cap_len)
        encoded = np.asarray(encoded)
        return encoded

    def get_image(self, index):
        if self.images is not None:
            return self.images[index]

        img_data = self.data.iloc[index, :]
        img = Image.open('CUB_200_2011/images/' + img_data.img_path).convert('RGB')
        width, height = img.size

        r = int(np.maximum(img_data.bbox_width, img_data.bbox_height) * 0.75)
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
                re_img = transforms.Resize(self.img_size[i])(img)
            else:
                re_img = img
            imgs.append(self.normalize(re_img))
        return imgs


class CUB:
    def __init__(self):
        self.collate_fn = collate_fn
        print('Loading image paths ...')
        self.data = pd.read_csv('CUB_200_2011/images.txt', delim_whitespace=True, header=None, index_col=0,
                                names=['img_path'])

        for i in range(CAPTIONS):
            self.data[f'caption_{i}'] = ''

        for i, row in self.data.iterrows():
            dir, file = row['img_path'].split('/')
            file = file.replace('.jpg', '.txt')

            captions = []
            # Keep only alphanumeric characters
            pattern = re.compile(r'[^\sa-z]+')
            with open(f'CUB_200_2011/text/{dir}/{file}', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.replace('\ufffd\ufffd', ' ').replace(',', ' ').replace('-', ' ').replace('/',
                                                                                                         ' ').lower().strip()
                    line = line.split()[:CAP_MAX_LEN]
                    line = ' '.join(line)
                    line = pattern.sub('', line)
                    captions.append(line)

            for j, c in enumerate(captions):
                row[f'caption_{j}'] = c

        print('Setting train/test split ...')
        with open('CUB_200_2011/test/filenames.pickle', 'rb') as f:
            test = pickle.load(f)

        self.data['train'] = self.data['img_path'].apply(lambda x: 0 if x[:x.index('.jpg')] in test else 1)
        self.data.sort_values(by='train', inplace=True, ascending=False)

        self.word_freq = count_word_freq(self.data[self.data.train == 1])
        self.vocab = build_vocab(self.word_freq)
        print(f'Vocab size: {len(self.vocab)}')

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

        print('Loading class labels ...')
        class_labels = pd.read_csv('CUB_200_2011/image_class_labels.txt', delim_whitespace=True, header=None,
                                   index_col=0, names=['class'])
        self.data = self.data.join(class_labels)

        self.train = CUBSubset(self.data[self.data.train == 1], self.vocab, self.imsize, self.transforms,
                               self.normalize, preload=False)
        self.test = CUBSubset(self.data[self.data.train == 0], self.vocab, self.imsize, self.transforms,
                              self.normalize, preload=False)


class FlowersSubset(Dataset):
    def __init__(self, subset, vocab, img_size, transformations, normalize):
        self.data = subset
        self.vocab = vocab
        self.img_size = img_size
        self.transforms = transformations
        self.normalize = normalize

    def __getitem__(self, index):
        imgs = self.get_image(index)
        caption = self.get_caption(index)
        label = self.data.iloc[index]['class']
        return imgs, caption, label

    def __len__(self):
        return self.data.shape[0]

    def get_image(self, index):
        img_path = self.data.iloc[index].img_path
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        size = min(width, height)
        img = transforms.RandomCrop(size)(img)
        img = self.transforms(img)

        imgs = []
        for i in range(BRANCH_NUM):
            if i < (BRANCH_NUM - 1):
                re_img = transforms.Resize(self.img_size[i])(img)
            else:
                re_img = img
            imgs.append(self.normalize(re_img))
        return imgs

    def get_caption(self, index):
        caption_idx = random.randint(0, CAPTIONS - 1)
        caption = self.data[f'caption_{caption_idx}'].iloc[index]
        return self.encode_text(caption)

    def encode_text(self, text):
        encoded = [self.vocab.get(w, self.vocab[UNK_TOKEN]) for w in text.split()]
        encoded = encoded[:CAP_MAX_LEN]  # address END token
        cap_len = len(encoded)
        encoded = encoded + [self.vocab[END_TOKEN]] * (CAP_MAX_LEN - cap_len)
        encoded = np.asarray(encoded)
        return encoded


class Flowers:
    def __init__(self):
        self.collate_fn = collate_fn
        print('Loading image paths ...')
        image_files = glob.glob('Flowers102/image/*.jpg')
        self.data = pd.DataFrame(image_files, columns=['img_path'])
        self.data['name'] = [f[-15:-4] for f in image_files]

        for i in range(CAPTIONS):
            self.data[f'caption_{i}'] = ''

        text_files = glob.glob('Flowers102/text/*/*.txt')

        print('Loading captions ...')
        self.data['class'] = 0
        # Keep only alphanumeric characters
        pattern = re.compile(r'[^\sa-z]+')
        for f in text_files:
            name = f[-15:-4]
            class_label = int(f[-21:-16])
            self.data.loc[self.data.name == name, 'class'] = class_label

            with open(f, 'r', encoding='utf-8') as captions:
                for i, line in enumerate(captions.readlines()):
                    line = line.replace(',', ' ').replace('-', ' ').replace('/', ' ').lower().strip()
                    line = line.split()[:CAP_MAX_LEN]
                    line = ' '.join(line)
                    line = pattern.sub('', line)
                    self.data.loc[self.data.name == name, f'caption_{i}'] = line

        print('Setting train/test split ...')
        self.data['train'] = 1
        with open('Flowers102/valclasses.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                class_id = int(line[6:-1])
                self.data.loc[self.data['class'] == class_id, 'train'] = 0

        self.word_freq = count_word_freq(self.data[self.data.train == 1])
        self.vocab = build_vocab(self.word_freq)
        print(f'Vocab size: {len(self.vocab)}')

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

        self.train = FlowersSubset(self.data[self.data.train == 1], self.vocab, self.imsize, self.transforms,
                                   self.normalize)

        self.test = FlowersSubset(self.data[self.data.train == 0], self.vocab, self.imsize, self.transforms,
                                  self.normalize)
