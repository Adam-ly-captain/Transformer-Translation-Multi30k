from .base import BaseDataset, BaseCollator, BaseDataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
import tqdm
import os
import re


class MMTDataset(BaseDataset):
    def __init__(self, data, dataset_type='train', **kwargs):
        super().__init__(dataset_type=dataset_type, **kwargs)
        self.data = data
        self.processed_data_path = kwargs.get("processed_data_path", "./data/multi30k/processed/")
        
        self.src_vocab, self.tgt_vocab = self.build_vocab()
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

    def __getitem__(self, index):
        src_sentence, tgt_sentence = self.data[index]
        if self.dataset_type == 'train':
            return self.convert_src_sentence_to_indices('<sos> ' + src_sentence), \
                    self.convert_tgt_sentence_to_indices('<sos> ' + tgt_sentence), \
                    self.convert_tgt_sentence_to_indices(tgt_sentence + ' <eos>')
        else:
            return self.convert_src_sentence_to_indices('<sos> ' + src_sentence), \
                    self.convert_tgt_sentence_to_indices('<sos>'), \
                    self.convert_tgt_sentence_to_indices(tgt_sentence + ' <eos>')

    def __len__(self):
        return len(self.data)
    
    def build_vocab(self):
        # training data, validation data and test data share the same vocabulary
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)

        src_vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        tgt_vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        if os.path.exists(f'{self.processed_data_path}/src_vocab.txt') and os.path.exists(f'{self.processed_data_path}/tgt_vocab.txt'):
            with open(f'{self.processed_data_path}/src_vocab.txt', 'r') as f:
                src_vocab = [line.strip() for line in f.readlines()]

            with open(f'{self.processed_data_path}/tgt_vocab.txt', 'r') as f:
                tgt_vocab = [line.strip() for line in f.readlines()]
            
            # shared vocabulary has been built
            return src_vocab, tgt_vocab
        
        if os.path.exists(f'{self.processed_data_path}/src_vocab_temp.txt') and os.path.exists(f'{self.processed_data_path}/tgt_vocab_temp.txt'):
            with open(f'{self.processed_data_path}/src_vocab_temp.txt', 'r') as f:
                src_vocab = [line.strip() for line in f.readlines()]
                
            with open(f'{self.processed_data_path}/tgt_vocab_temp.txt', 'r') as f:
                tgt_vocab = [line.strip() for line in f.readlines()]

        src_vocab_set = set(src_vocab)
        tgt_vocab_set = set(tgt_vocab)
        for sentences in tqdm.tqdm(self.data, desc=f"Building {self.dataset_type} vocabularies"):
            src, tgt = sentences
            for word in src.split():
                if word not in src_vocab_set:
                    src_vocab.append(re.sub(r'(?<!<)[^\w\s<>](?!>)', '', word).lower())
                    src_vocab_set.add(word)

            for word in tgt.split():
                if word not in tgt_vocab_set:
                    tgt_vocab.append(re.sub(r'(?<!<)[^\w\s<>](?!>)', '', word).lower())
                    tgt_vocab_set.add(word)

        with open(f'{self.processed_data_path}/src_vocab_temp.txt', 'w') as f:
            for word in src_vocab:
                f.write(word + '\n')

        with open(f'{self.processed_data_path}/tgt_vocab_temp.txt', 'w') as f:
            for word in tgt_vocab:
                f.write(word + '\n')

        del src_vocab_set
        del tgt_vocab_set

        return src_vocab, tgt_vocab
    
    def get_src_vocab_size(self):
        return len(self.src_vocab)

    def get_tgt_vocab_size(self):
        return len(self.tgt_vocab)

    def convert_src_sentence_to_indices(self, sentence):
        indices = []
        for word in sentence.split():
            word = re.sub(r'(?<!<)[^\w\s<>](?!>)', '', word).lower()
            if word in self.src_vocab:
                indices.append(self.src_vocab.index(word))
            else:
                indices.append(self.src_vocab.index('<unk>'))
        return indices

    def convert_tgt_sentence_to_indices(self, sentence):
        indices = []
        for word in sentence.split():
            word = re.sub(r'(?<!<)[^\w\s<>](?!>)', '', word).lower()
            if word in self.tgt_vocab:
                indices.append(self.tgt_vocab.index(word))
            else:
                indices.append(self.tgt_vocab.index('<unk>'))
        return indices

    def get_sos_idx(self):
        return self.src_vocab.index('<sos>')

    def get_pad_idx(self):
        return self.src_vocab.index('<pad>')
    
    def get_eos_idx(self):
        return self.tgt_vocab.index('<eos>')

    def update_vocab(self):
        if os.path.exists(f'{self.processed_data_path}/src_vocab.txt') and os.path.exists(f'{self.processed_data_path}/tgt_vocab.txt'):
             # vocab already updated
            with open(f'{self.processed_data_path}/src_vocab.txt', 'r') as f:
                src_vocab = [line.strip() for line in f.readlines()]

            with open(f'{self.processed_data_path}/tgt_vocab.txt', 'r') as f:
                tgt_vocab = [line.strip() for line in f.readlines()]

            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            return
        
        if os.path.exists(f'{self.processed_data_path}/src_vocab_temp.txt') and os.path.exists(f'{self.processed_data_path}/tgt_vocab_temp.txt'):
            with open(f'{self.processed_data_path}/src_vocab_temp.txt', 'r') as f:
                src_vocab = [line.strip() for line in f.readlines()]

            with open(f'{self.processed_data_path}/tgt_vocab_temp.txt', 'r') as f:
                tgt_vocab = [line.strip() for line in f.readlines()]
        
        with open(f'{self.processed_data_path}/src_vocab.txt', 'w') as f:
            for word in src_vocab:
                f.write(word + '\n')

        with open(f'{self.processed_data_path}/tgt_vocab.txt', 'w') as f:
            for word in tgt_vocab:
                f.write(word + '\n')
        
        # delete temporary vocab files
        os.remove(f'{self.processed_data_path}/src_vocab_temp.txt')
        os.remove(f'{self.processed_data_path}/tgt_vocab_temp.txt')
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


class MMTCollator(BaseCollator):
    def __init__(self, dataset_type, pad_idx=0, eos_idx=0):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.dataset_type = dataset_type

    def __call__(self, batch):
        src_batch, tgt_shift_batch, tgt_batch = zip(*batch)
        
        # Pad source sequences
        src_lengths = [len(seq) for seq in src_batch]
        max_src_length = max(src_lengths)
        padded_src_batch = [seq + [self.pad_idx] * (max_src_length - len(seq)) for seq in src_batch]
        
        # Pad target sequences
        tgt_lengths = [len(seq) for seq in tgt_batch]
        max_tgt_length = max(tgt_lengths)
        padded_tgt_batch = [seq + [self.pad_idx] * (max_tgt_length - len(seq)) for seq in tgt_batch]
        
        if self.dataset_type == 'train':
            # Pad target shift sequences
            padded_tgt_shift_batch = [seq + [self.pad_idx] * (max_tgt_length - len(seq)) for seq in tgt_shift_batch]
            
            return torch.tensor(padded_src_batch), torch.tensor(padded_tgt_shift_batch), torch.tensor(padded_tgt_batch)
        else:
            return torch.tensor(padded_src_batch), torch.tensor(tgt_shift_batch), torch.tensor(padded_tgt_batch)


class MMTDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=False,
                 num_workers=1, collate_fn=None, **kwargs):
        super().__init__(dataset=dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=collate_fn or MMTCollator(dataset_type=dataset.get_dataset_type(), pad_idx=dataset.get_pad_idx()), **kwargs)

        # shared vocabulary update
        self.dataset.update_vocab()
        self.log("src vocab size: {}, tgt vocab size: {}".format(dataset.get_src_vocab_size(), dataset.get_tgt_vocab_size()))

    def get_dataset_config(self):
        return {
            "src_vocab_size": self.dataset.get_src_vocab_size(),
            "tgt_vocab_size": self.dataset.get_tgt_vocab_size(),
            "sos_idx": self.dataset.get_sos_idx(),
            "pad_idx": self.dataset.get_pad_idx(),
            "eos_idx": self.dataset.get_eos_idx()
        }
        