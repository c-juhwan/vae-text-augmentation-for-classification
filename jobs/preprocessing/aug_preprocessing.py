# Standard Library Modules
import os
import sys
import pickle
import argparse
from collections import Counter, OrderedDict
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from nltk.tokenize import WordPunctTokenizer, sent_tokenize
# Pytorch Modules
import torch
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import check_path

def load_data(dataset_path:str) -> tuple: # (pd.DataFrame, pd.DataFrame, int)
    """
    Load data from given path.
    """
    # Load csv file
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'), sep=',', encoding='utf-8')
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'), sep=',', encoding='utf-8')

    # Load classes.txt
    with open(os.path.join(dataset_path, 'classes.txt'), 'r') as f:
        classes = f.read().splitlines()
    num_classes = len(classes)

    return train_df, test_df, num_classes

def preprocessing(args:argparse.Namespace) -> None:
    # Load data
    train_df, test_df, num_classes = load_data(os.path.join(args.data_path, args.task_dataset))
    # Shuffle train_df and create valid_df
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    valid_df = train_df.iloc[:int(len(train_df)*0.1)]
    valid_df = valid_df.reset_index(drop=True)
    train_df = train_df.iloc[int(len(train_df)*0.1):]
    train_df = train_df.reset_index(drop=True)

    # Build vocabulary
    sent_tokenizer = get_tokenizer(sent_tokenize)
    tokenizer = get_tokenizer(WordPunctTokenizer().tokenize)
    counter = Counter()
    for idx in tqdm(range(len(train_df)), desc='Building Vocab'):
        text = train_df['text'][idx]
        counter.update(tokenizer(text))
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulary = vocab(ordered_dict, specials=['<pad>', '<unk>', '<bos>', '<eos>'], min_freq=args.vocab_min_freq)
    vocabulary.set_default_index(vocabulary['<unk>'])
    vocab_size = len(vocabulary)
    print(f"vocab_size: {vocab_size}")

    # Preprocessing
    data_dict = {
        'train' : {
            'Text' : [],
            'Label' : [],
            'Num_Label' : num_classes,
            'Vocab' : vocabulary,
            'Vocab_Size' : vocab_size
        },
        'valid' : {
            'Text' : [],
            'Label' : [],
            'Num_Label' : num_classes,
            'Vocab' : vocabulary,
            'Vocab_Size' : vocab_size
        },
        'test' : {
            'Text' : [],
            'Label' : [],
            'Num_Label' : num_classes,
            'Vocab' : vocabulary,
            'Vocab_Size' : vocab_size
        },
    }

    for split_df, split in zip([train_df, valid_df, test_df], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_df)), desc=f'Preprocessing for {split}'):
            text = split_df['text'][idx]
            label = split_df['class'][idx]

            # Tokenize
            tokenized_text = []
            for each_sent in sent_tokenizer(text):
                tokenized_sent = tokenizer(each_sent)
                tokenized_text.extend(tokenized_sent)
            # Convert to index
            indexed_text = [vocabulary[token] for token in tokenized_text]
            # Add special tokens
            indexed_text = [vocabulary['<bos>']] + indexed_text + [vocabulary['<eos>']]
            if len(indexed_text) < args.max_seq_len: # Add padding
                indexed_text += [vocabulary['<pad>']] * (args.max_seq_len - len(indexed_text))
            else: # Truncate
                indexed_text = indexed_text[:args.max_seq_len]
            # Convert to tensor
            indexed_text = torch.tensor(indexed_text, dtype=torch.long)

            data_dict[split]['Text'].append(indexed_text)
            data_dict[split]['Label'].append(torch.tensor(label, dtype=torch.long)) # Cross Entropy Loss requires LongTensor

    # Save data as pickle file
    check_path(os.path.join(args.preprocessed_path, args.task, args.task_dataset))
    with open(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'train_{args.max_seq_len}.pkl'), 'wb') as f:
        pickle.dump(data_dict['train'], f)
    with open(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'valid_{args.max_seq_len}.pkl'), 'wb') as f:
        pickle.dump(data_dict['valid'], f)
    with open(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'test_{args.max_seq_len}.pkl'), 'wb') as f:
        pickle.dump(data_dict['test'], f)
