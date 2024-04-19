#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''

# fixed UnicodeDecodeError: 'gbk' codec can't decode byte 0x99 in position 264:
# by adding open(...,'r', encoding='UTF-8')

import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
from random import randint, sample


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

# Mean pooling prompt for SST
class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args, tokenizer):
        self.dataset = dataset
        self.p = args
        self.tokenizer = tokenizer
    
        self.example_sentences = [
            "Suffers from the lack of a compelling or comprehensible narrative.",
            "You wo n't like Roger , but you will quickly recognize him.",
            "No one goes unindicted here , which is probably for the best.",
            "It 's a lovely film with lovely performances by Buy and Accorsi.",
            "A warm , funny , engaging film."
        ]
        
        self.label_words = ['terrible', 'bad', 'neutral', 'good', 'excellent']
        self.dataset = [(self.format_sentence(sentence), label, sent_id) for sentence, label, sent_id in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def format_sentence(self, original_sentence):
        prompt = f"{original_sentence} it is [MASK]. "
        for example, label_word in zip(self.example_sentences, self.label_words):
            prompt += f"{example} it is {label_word}."
        return prompt

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids
    
    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids,
        }
        

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args, tokenizer):
        self.dataset = dataset
        self.p = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    
    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_98tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids,
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, tokenizer, isRegression =False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args, tokenizer):
        self.dataset = dataset
        self.p = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data



def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data


from typing import List
from torch import Tensor

class Wiki1MDataset(Dataset):
    def __init__(self, path: str, args, tokenizer, max_length=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                if line:
                    self.data.append(line)

    def __getitem__(self, index: int) -> str:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, all_data: List[str]) -> dict:
        encoding = self.tokenizer(all_data, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        token_ids = Tensor(encoding['input_ids'])
        attention_mask = Tensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
        }

        return batched_data

class NLIDataset(Dataset):
    def __init__(self, filename, args, tokenizer):
        self.data = self.load_data(filename)
        self.p = args
        self.tokenizer = tokenizer

    def load_data(self, filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter=','):  # 使用逗号作为分隔符
                sent0 = record.get('sent0', '').strip()
                sent1 = record.get('sent1', '').strip()
                hard_neg = record.get('hard_neg', '').strip()
                data.append((sent0, sent1, hard_neg))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pad_data(self, data):
        sent0 = [x[0] for x in data]
        sent1 = [x[1] for x in data]
        hard_neg = [x[2] for x in data]

        encoding0 = self.tokenizer(sent0, return_tensors='pt', padding=True, truncation=True)
        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding_neg = self.tokenizer(hard_neg, return_tensors='pt', padding=True, truncation=True)

        return (encoding0['input_ids'], encoding0['attention_mask'],
                encoding1['input_ids'], encoding1['attention_mask'], 
                encoding_neg['input_ids'], encoding_neg['attention_mask'])

    def collate_fn(self, all_data):
        (token_ids_0, attention_mask_0,
         token_ids_1, attention_mask_1,
         token_ids_neg, attention_mask_neg) = self.pad_data(all_data)

        batched_data = {
            'token_ids_0': token_ids_0,
            'attention_mask_0': attention_mask_0,
            'token_ids_1': token_ids_1,
            'attention_mask_1': attention_mask_1,
            'token_ids_neg': token_ids_neg,
            'attention_mask_neg': attention_mask_neg,
        }

        return batched_data



class FP75agreeDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.data = self.load_data(filename)
        self.tokenizer = tokenizer

    def load_data(self, filename):
        data = []
        with open(filename, 'r', encoding='ISO-8859-1') as fp:
            for line in fp:
                # Splitting the line into the sentence and its label
                sent, label = line.strip().split('@')
                data.append((sent, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def label_mapping_function(self, labels):
        """
        Maps text labels to integers.
        
        Args:
        - labels (list of str): List of textual labels.
        
        Returns:
        - torch.Tensor: Tensor of mapped integer labels.
        """
        mapping = {
            "positive": 0,
            "neutral": 1,
            "negative": 2
        }
        mapped_labels = [mapping[label] for label in labels]
        return torch.tensor(mapped_labels)
    
    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        
        labels = self.label_mapping_function(labels) # "positive" -> 0, etc.

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)

        return (encoding['input_ids'], encoding['attention_mask'], labels)

    def collate_fn(self, all_data):
        (token_ids, attention_mask, labels) = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return batched_data

import string

class MSDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.data = self.load_data(filename)
        self.tokenizer = tokenizer
        self.label_map = {'positive': 1, 'negative': -1, 'neutral': 0}

    def preprocess_text(self, text):
        # 转为小写
        text = text.lower()

        # 移除标点符号
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        return text
    
    def load_data(self, filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for record in reader:
                text = record['text'].strip()
                text = self.preprocess_text(text)
                label = record['label'].strip()
                data.append((text, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        
        labels = torch.tensor([self.label_map[label] for label in labels], dtype=torch.long)
        
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)

        return (encoding['input_ids'], encoding['attention_mask'], labels)

    def collate_fn(self, all_data):
        (token_ids, attention_mask, labels) = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return batched_data