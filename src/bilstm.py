import os
import sys
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import os
import pickle as pkl
import pandas as pd
from itertools import chain
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from model import LSTMNet, train_model, eval_model, pred_model
from sklearn.model_selection import KFold
import pdb

DATA_ROOT = '../input'

class config:
    training_mode = False
    batch_size = 256
    learning_rate = 0.01
    regulation = 0.00001
    epochs = 30
    checkpoints = ['LSTMNet_0.928fold0.pth', 'LSTMNet_0.929fold1.pth', 'LSTMNet_0.931fold2.pth', 'LSTMNet_0.928fold3.pth', 'LSTMNet_0.927fold4.pth']
    kfold = 5

train_dict = pkl.load(open(osp.join(DATA_ROOT, "train.pkl"), "rb"))
val_dict = pkl.load(open(osp.join(DATA_ROOT, "val.pkl"), "rb"))
test_dict = pkl.load(open(osp.join(DATA_ROOT, "test.pkl"), "rb"))

trainval_dict = dict()
trainval_dict['id'] = np.array(train_dict['id'] + [x + len(train_dict['id']) for x in val_dict['id']])
trainval_dict['word_seq'] = np.array(train_dict['word_seq'] + val_dict['word_seq'])
trainval_dict['tag_seq'] = np.array(train_dict['tag_seq'] + val_dict['tag_seq'])

vocab_dict = {'_unk_': 0, '_w_pad_': 1}

for doc in train_dict['word_seq'] + val_dict['word_seq'] + test_dict['word_seq']:
    for word in doc:
        if(word not in vocab_dict):
            vocab_dict[word] = len(vocab_dict)

tag_dict = {'_t_pad_': 0} # add a padding token

for tag_seq in train_dict['tag_seq'] + val_dict['tag_seq']:
    for tag in tag_seq:
        if(tag not in tag_dict):
            tag_dict[tag] = len(tag_dict)

word2idx = vocab_dict
idx2word = {v:k for k,v in word2idx.items()}
tag2idx = tag_dict
idx2tag = {v:k for k,v in tag2idx.items()}

max_sent_length = 128

def slice_data(trainval_dict, train_idx, val_idx):
    train_d = dict()
    val_d = dict()
    for k in ['word_seq', 'tag_seq']:
        train_d[k] = trainval_dict[k][train_idx]
        val_d[k] = trainval_dict[k][val_idx]

    return train_d, val_d

def torch_data(train_dict, val_dict, test_dict):
    train_tokens = np.array([[word2idx[w] for w in doc] for doc in train_dict['word_seq']])
    val_tokens = np.array([[word2idx.get(w, 0) for w in doc] for doc in val_dict['word_seq']])
    test_tokens = np.array([[word2idx.get(w, 0) for w in doc] for doc in test_dict['word_seq']])

    train_tags = np.array([[tag2idx[t] for t in t_seq] for t_seq in train_dict['tag_seq']])

    val_tags = np.array([[tag2idx[t] for t in t_seq] for t_seq in val_dict['tag_seq']])

    train_dataset = TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_tags))
    val_dataset = TensorDataset(torch.from_numpy(val_tokens), torch.from_numpy(val_tags))
    test_dataset = TensorDataset(torch.from_numpy(test_tokens), torch.zeros(test_tokens.shape))

    return {'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8),
            'valid': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8),
            'test': DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)}


def get_pred_csv(pred, f_dict, padding_id="_w_pad_"):
    ids = [str(i)+"_"+str(w_id) for i, _ in enumerate(f_dict["id"]) for w_id, word in enumerate(f_dict["word_seq"][i]) if word != padding_id]
    tags = [ps[w_id] for i, ps in enumerate(pred) for w_id, word in enumerate(f_dict["word_seq"][i]) if word != padding_id]
    return {"id":ids, "tag":tags}

def get_csv_accuracy(d1, d2):
    assert len(d1["id"]) == len(d2["id"])
    return sum(np.array(d1["tag"]) == np.array(d2["tag"])) / len(d1["tag"])

kf = KFold(n_splits=config.kfold, random_state=2021)
if config.training_mode:
    for i, (train_idx, val_idx) in enumerate(kf.split(range(len(trainval_dict['id'])))):
        train_d, val_d = slice_data(trainval_dict, train_idx, val_idx)
        dataloaders = torch_data(train_d, val_d, test_dict)
        model = LSTMNet(len(vocab_dict), len(tag_dict))
        print('################ Fold {} ################'.format(i))
        train_model(model, dataloaders, 
                    learn_rate=config.learning_rate,
                    regulation=config.regulation,
                    fold=i,
                    num_epoch=config.epochs)

else:
    dataloaders = torch_data(train_dict, val_dict, test_dict)
    model = LSTMNet(len(vocab_dict), len(tag_dict))
    preds = pred_model(model, dataloaders, config.checkpoints)
    preds = np.array([[idx2tag[p] for p in ps] for ps in preds])
    pred_dict = get_pred_csv(preds, test_dict, padding_id="_w_pad_")
    pd.DataFrame(pred_dict).to_csv("test_pred.csv", index=False)
