import pandas as pd
import random
import torch
from tqdm import tqdm

#create vocabulary from dataset
def get_vocabulary(train_texts, **params):
    token_counts = {}
    vocabulary = {}
    token_id = 0

    for text in train_texts:
        tokens = list(text.strip()) if params['characters'] else text.strip().split()
        for token in tokens:
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1

    for token, count in token_counts.items():
        if count >= params['min_count']:
            vocabulary[token] = token_id
            token_id += 1

    for special_token in [params['start_token'], params['end_token'], params['unk_token']]:
        if special_token not in vocabulary:
            vocabulary[special_token] = token_id
            token_id += 1

    return vocabulary

#split the data into test and train
def split_test_data(input_data, **params):
    test_size = params['testsize']
    grouped = input_data.groupby('language')

    train_data = []
    test_data = []

    for language, group in grouped:
        train_data.append(group.iloc[test_size:])
        test_data.append(group.iloc[:test_size])
    
    return pd.concat(train_data).reset_index(drop=True), pd.concat(test_data).reset_index(drop=True)


def process_data(text, targets=None, cv=False, **kwargs):
    max_length = kwargs['max_length']
    max_words = min(max_length, max(len(t.strip()) for t in text))
    vocab = kwargs['vocab']
    vocab_size = len(vocab)
    unk_token = kwargs['unk_token']
    utoken_value = vocab.get(unk_token)
    cv_percentage = kwargs.get('cv_percentage', 0.1)
    verbose = kwargs.get('verbose', 1)

    non_empty_lines = [l.strip() for l in text if l.strip()]
    nums = len(non_empty_lines)
    
    dataset = torch.full((max_words, nums), vocab_size, dtype=torch.long)
    labels = torch.zeros(nums, dtype=torch.uint8)
    
    lengths = []
    index = 0
    for i, line in tqdm(enumerate(text), desc='Allocating data memory', disable=verbose < 2):
        words = list(line.strip()) if kwargs['characters'] else line.strip().split()
        if not words:
            continue
        truncated_words = words[:max_words]
        lengths.append(len(truncated_words))
        
        for index_2, word in enumerate(truncated_words):
            dataset[index_2, index] = vocab.get(word, utoken_value)
        
        if targets is not None:
            labels[index] = kwargs['outputs'][targets[i]]
        
        index += 1
    
    assert index == nums
    
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    if not cv:
        return dataset, lengths, labels

    indexes = list(range(nums))
    random.shuffle(indexes)
    split_point = int(nums * (1 - cv_percentage))
    
    train_indices = indexes[:split_point]
    valid_indices = indexes[split_point:]
    
    train_set = dataset[:, train_indices]
    train_labels = labels[train_indices]
    valid_set = dataset[:, valid_indices]
    valid_labels = labels[valid_indices]
    train_lengths = lengths[train_indices]
    valid_lengths = lengths[valid_indices]
    
    return train_set, train_lengths, valid_set, valid_lengths, train_labels, valid_labels

