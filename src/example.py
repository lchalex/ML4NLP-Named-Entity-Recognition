import os
import sys
import pdb
import torch
import torch.nn as nn
import keras
from keras.utils import to_categorical
import numpy as np
import os
import pickle as pkl
import pandas as pd
from itertools import chain
import pdb

train_dict = pkl.load(open("../input/train.pkl", "rb"))
val_dict = pkl.load(open("../input/val.pkl", "rb"))
test_dict = pkl.load(open("../input/test.pkl", "rb"))
print("keys in train_dict:", train_dict.keys())
print("keys in val_dict:", val_dict.keys())
print("keys in test_dict:", test_dict.keys())

print("index:", train_dict["id"][0])
print(*zip(train_dict["word_seq"][0], train_dict["tag_seq"][0]))

print("count of the NER tags:", len(set(chain(*train_dict["tag_seq"]))))
print("all the NER tags:", set(chain(*train_dict["tag_seq"])))

vocab_dict = {'_unk_': 0, '_w_pad_': 1}

for doc in train_dict['word_seq']:
    for word in doc:
        if(word not in vocab_dict):
            vocab_dict[word] = len(vocab_dict)

tag_dict = {'_t_pad_': 0} # add a padding token

for tag_seq in train_dict['tag_seq']:
    for tag in tag_seq:
        if(tag not in tag_dict):
            tag_dict[tag] = len(tag_dict)
word2idx = vocab_dict
idx2word = {v:k for k,v in word2idx.items()}
tag2idx = tag_dict
idx2tag = {v:k for k,v in tag2idx.items()}            

print("size of word vocab:", len(vocab_dict), "size of tag_dict:", len(tag_dict))

# The maximum length of a sentence is set to 128
max_sent_length = 128

train_tokens = np.array([[word2idx[w] for w in doc] for doc in train_dict['word_seq']])
val_tokens = np.array([[word2idx.get(w, 0) for w in doc] for doc in val_dict['word_seq']])
test_tokens = np.array([[word2idx.get(w, 0) for w in doc] for doc in test_dict['word_seq']])


train_tags = [[tag2idx[t] for t in t_seq] for t_seq in train_dict['tag_seq']]
train_tags = np.array([to_categorical(t_seq, num_classes=len(tag_dict)) for t_seq in train_tags])

val_tags = [[tag2idx[t] for t in t_seq] for t_seq in val_dict['tag_seq']]
val_tags = np.array([to_categorical(t_seq, num_classes=len(tag_dict)) for t_seq in val_tags])

print("training size:", train_tokens.shape, "tag size:", train_tags.shape)
print("validating size:", val_tokens.shape, "tag size:", val_tags.shape)


print(train_tokens[0,:10], np.argmax(train_tags[0, :10, :], axis=1))

def get_pred_csv(pred, f_dict, padding_id="_w_pad_"):
    ids = [str(i)+"_"+str(w_id) for i, _ in enumerate(f_dict["id"]) for w_id, word in enumerate(f_dict["word_seq"][i]) if word != padding_id]
    tags = [ps[w_id] for i, ps in enumerate(pred) for w_id, word in enumerate(f_dict["word_seq"][i]) if word != padding_id]
    return {"id":ids, "tag":tags}

def get_csv_accuracy(d1, d2):
    assert len(d1["id"]) == len(d2["id"])
    return sum(np.array(d1["tag"]) == np.array(d2["tag"])) / len(d1["tag"])

val_preds = np.array([[idx2tag[p] for p in preds] for preds in np.ones((len(val_dict["id"]), max_sent_length))])
val_pred_dict = get_pred_csv(val_preds, val_dict)
print("validation acc", get_csv_accuracy(val_pred_dict, pd.read_csv("../input/val_labels.csv")))

test_preds = np.array([[idx2tag[p] for p in preds] for preds in np.ones((len(test_dict["id"]), max_sent_length))])

test_preds_dict = get_pred_csv(test_preds, test_dict)
pd.DataFrame(test_preds_dict).to_csv("sample_submission.csv", index=False)

# Please submit this file
pd.read_csv("sample_submission.csv")

# 1. Predict all the tags as "O"
val_preds = np.array([[idx2tag[p] for p in preds] for preds in np.ones((len(val_dict["id"]), max_sent_length))])

val_pred_dict = get_pred_csv(val_preds, val_dict)

pd.DataFrame(val_pred_dict).to_csv("val_pred.csv", index=False)
print("validation acc", 
      get_csv_accuracy(pd.read_csv("val_pred.csv"), 
                       pd.read_csv("../input/val_labels.csv")))
