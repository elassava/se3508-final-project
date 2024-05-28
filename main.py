import argparse
from data_helpers import get_vocabulary, process_data, split_test_data
from model_helpers import model_train, model_evaluate, model_save
from language_predictor import determine_language
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

#argument parser setup
parser = argparse.ArgumentParser(description='Language Prediction Model')
parser.add_argument('--test', action='store_true', help='If set, the model will only be tested. Otherwise, it will be trained.')
args = parser.parse_args()

#arguments dictionary to pass the arguments easier
params = {
    'input_file': None,
    'vocabulary': None,
    'cv_percentage': 0.1,
    'epochs': 10,
    'batch_size': 32,
    'embedding_size': 128,
    'hidden_size': 512,
    'num_layers': 1,
    'learning_rate': 0.001,
    'seed': 0,
    'start_token': '<s>',
    'end_token': '<\\s>',
    'unk_token': '<UNK>',
    'verbose': 1,
    'characters': True,
    'min_count': 50,
    'max_length': 300,
    'testsize': 100,
    'device': torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
}

#model paths
encoder_path = "saved_models/encoder.pth"
predictor_path = "saved_models/predictor.pth"

#data preparation
input_df = pd.read_csv('data/dataset.csv')
train_df, test_df = split_test_data(input_df, **params)
params['vocab'] = get_vocabulary(train_df.Text.values, **params)
params['outputs'] = {t: i for i, t in enumerate(np.unique(train_df.language.values))}

#model definitions
class LanguageEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(LanguageEncoder, self).__init__()
        self.vocab = kwargs['vocab']
        self.in_dim = len(self.vocab)
        self.embed_dim = kwargs['embedding_size']
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']

        self.embed = nn.Embedding(self.in_dim + 1, self.embed_dim, padding_idx=self.in_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hid_dim, num_layers=self.n_layers)

    def forward(self, inputs, lengths):
        emb = self.embed(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, enforce_sorted=False)
        packed_rec, self.hidden = self.lstm(packed, self.hidden)
        rec, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rec)
        out = rec[lengths - 1, list(range(rec.shape[1])), :]
        return out

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        self.hidden = (weight.new_zeros(self.n_layers, bsz, self.hid_dim),
                       weight.new_zeros(self.n_layers, bsz, self.hid_dim))

    def detach_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def cpu_hidden(self):
        self.hidden = (self.hidden[0].detach().cpu(), self.hidden[1].detach().cpu())

class LanguagePredictor(nn.Module):
    def __init__(self, **kwargs):
        super(LanguagePredictor, self).__init__()
        self.hid_dim = kwargs['hidden_size']
        self.out_dim = len(kwargs['outputs'])
        self.linear = nn.Linear(self.hid_dim, self.out_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        return self.softmax(self.linear(inputs))

#function to load existing model
def load_existing_model():
    print("Loading model...")
    if os.path.exists(encoder_path) and os.path.exists(predictor_path):
        encoder.load_state_dict(torch.load(encoder_path))
        predictor.load_state_dict(torch.load(predictor_path))
        print("Model loaded successfully.")
        return encoder, predictor
    else:
        print("Model not found. Please train the model first.")
        return None, None

#train function
def start_training():
    print('Training...')
    losses = []
    for ep in tqdm(range(1, params['epochs'] + 1)):
        loss = model_train(trainset, trainlabels, trainlengths, encoder, predictor, optimizer, criterion, **params)
        val_pred = model_evaluate(validset, validlengths, encoder, predictor, **params)
        acc = 100 * len(np.where(val_pred == validlabels.numpy())[0]) / validset.shape[1]
        print(f'Epoch {ep} of {params["epochs"]}. Training loss: {loss:.2f}, Accuracy: {acc:.2f}%') #print loss and accuracy
        losses.append(loss)
    model_save(encoder, predictor) #save the trained model

#function for testing
def start_testing():
    print("Testing model...")
    input_text = input("Enter text: ")
    predicted_language = determine_language(input_text, encoder, predictor, params)
    print("Predicted language:", predicted_language)


#model initialization
encoder = LanguageEncoder(**params).to(params['device'])
predictor = LanguagePredictor(**params).to(params['device'])
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=params['learning_rate'])
criterion = nn.NLLLoss(reduction='mean').to(params['device'])

#data loading
trainset, trainlengths, validset, validlengths, trainlabels, validlabels = process_data(train_df.Text.values, targets=train_df.language.values, cv=True, **params)

if args.test:
    encoder, predictor = load_existing_model()
    start_testing()
else:
    start_training()
