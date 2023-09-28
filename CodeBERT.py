path='../config_dataset/data/qt/cc2vec/qt_train.pkl'

import pickle
import os
import sys
import torch.nn as nn

import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import BartModel,BartTokenizer,BartConfig,AutoModelForSequenceClassification,AutoTokenizer,RobertaModel,RobertaTokenizer,RobertaConfig


from untils import CustomDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import math
import os, torch
import random
import csv
import argparse
import time
batch_size = 16
from sklearn.metrics import roc_auc_score

class BART4JIT(nn.Module):
    def __init__(self, args):
        super(BART4JIT, self).__init__()
        self.args = args


        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # others
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(Dim, Co * 3)


        # CodeBERT model
        self.sentence_encoder =RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=args.class_num)




    def forward(self, msg_input_ids, msg_input_mask, code_input_ids, code_input_mask,
                ):

        # --------CodeBERT for msg------------
        msg_encoded = list()
        msg_encoded.append(self.sentence_encoder(input_ids=msg_input_ids, attention_mask=msg_input_mask)[1])
        msg = msg_encoded[0]
        # msg = msg_encoded[0]
        x_msg = self.fc3(msg)

        ##----- for CodeBERT code part----
        code_input_ids = code_input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)

        code_input_mask = code_input_mask.permute(1, 0, 2)

        num_file = 0
        x_encoded = []
        for i0 in range(len(code_input_ids)):  # tracerse all sentence
            num_file += 1
            x_encoded.append(self.sentence_encoder(input_ids=code_input_ids[i0], attention_mask=code_input_mask[i0])[1])
        x = torch.stack(x_encoded)
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)
        # CNN
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_code_line]
        # max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences) * ks
        x = torch.cat(x, 1)  # (batch_size, channel_output * ks)
        x_code = x

        ## Concatenate Code and Msg

        x_commit = torch.cat((x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)

        return out
def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')
    parser.add_argument('-valid', action='store_true')
    parser.add_argument('-train_data', type=str, help='the directory of our training data')
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')
    parser.add_argument('-weight', action='store_true', help='training DeepJIT model')
    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=4, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=128, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=768, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=3, help='the number of epochs')
    parser.add_argument('-save-dir', type=str, default='p_tuning_roberta/test/qt', help='where to save the snapshot')
    parser.add_argument('-zero_shot', action='store_true', help='zero shot learning')


    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    return parser


##最长512

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data and label at the given index
        data_point = self.data[index]
        label = self.labels[index]

        # You can do any preprocessing or transformations on the data here if needed

        # Return the data and label as tensors
        return torch.tensor(data_point), torch.tensor(label)


def convert_examples_to_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Tokenization and padding for one sentence ( for commit msg )
    """
    print(" max length at tokenizer:", max_seq_length)
    features_inputs = []
    features_masks = []
    features_segments = []
    index = 0
    for (ex_index, example) in enumerate(examples):

        # if index>10:
        #     break
        # index+=1

        tokens="[CLS] "+example+" [SEP]"


        tmp_result = tokenizer(tokens, padding='max_length', truncation=True, max_length=max_seq_length)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        input_ids=tmp_result['input_ids']
        input_mask=tmp_result['attention_mask']
        segment_ids=[0]*len(input_ids)


        features_inputs.append(input_ids)
        features_masks.append(input_mask)
        features_segments.append(segment_ids)

    return (features_inputs, features_masks, features_segments)


def convert_examples_to_hierarchical_features(examples, max_seq_length, tokenizer, params, print_examples=False):
    """
    Tokenization and padding for several sentence ( for commit code change; as code change consists of code lines from several code files)
    """

    features_inputs = []
    features_masks = []
    features_segments = []
    print(" max length for code tokenizer:", max_seq_length)
    index=0
    for (ex_index, example) in enumerate(examples):

        # if index>10:
        #     break
        # index+=1
        tokens_a = list()
        num_file = 0
        # print('num of code files:', len(example.split(" SEPARATOR_FOR_SENTENCE ")))
        for line in example.split(" SEPARATOR_FOR_SENTENCE "):

            if len(line.strip()) == 0:
                continue
            else:
                tokens_a.append('[CLS] '+line+' [SEP]')
                num_file += 1
            if num_file >= 4:
                break
        while(num_file<4):
            tokens_a.append("empty patch")
            num_file+=1
        # print("------------num of files are affectd:", num_file)
        # Account for [CLS] and [SEP]


        tokens = tokens_a
        segment_ids = list()

        # print("-------------------------------------------------------------------This is one tokenization---------------------")
        input_ids = list()
        input_mask = list()
        i = 0
        for line in tokens:
            # print("--------line before tokenizer------:", len(line))

            tmp_result = tokenizer(line, max_length=max_seq_length,padding='max_length', truncation=True)
            input_ids.append(tmp_result['input_ids'])
            input_mask.append(tmp_result['attention_mask'])
            segment_ids.append([0] * len(tmp_result['input_ids']))

            # print(i)
            i = i + 1
            # print("--------line after tokenizer------:", len(input_ids),len(tokens) ) #len(tokenizer.convert_tokens_to_ids(line)) )

        features_inputs.append(input_ids)
        features_masks.append(input_mask)
        features_segments.append(segment_ids)
    print(len(features_inputs), len(features_masks), len(features_segments))
    return (features_inputs, features_masks, features_segments)


def tokenization_for_codebert(data, max_length, flag, params):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    if flag == 'msg':
        features = convert_examples_to_features(examples=data, max_seq_length=max_length, tokenizer=tokenizer)
        return features
    elif flag == 'code':
        features = convert_examples_to_hierarchical_features(examples=data, max_seq_length=max_length,
                                                             tokenizer=tokenizer, params=params)
        return features
    else:
        print(" the flag is wrong for the tokenization of CodeBERT ")

def write_csv(path_dir,file_name, data):
    print('save loss in file: ', file_name)
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
    save_path = os.path.join(path_dir, file_name)
    if not os.path.exists(save_path):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    else:
        with open(save_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

def save(model, save_dir, save_prefix, epochs, step_, step):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}_{}_{}.pt'.format(save_prefix, epochs, step_, step)
    print('path:', save_path)
    torch.save(model.state_dict(), save_path)


def mini_batches(X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids, Y,
                 mini_batch_size=64, seed=0):
    ''' for testing; put every data into it
    '''

    m = Y.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_msg_input_ids, shuffled_X_msg_masks, shuffled_X_msg_segment_ids, shuffled_X_code_input_ids, shuffled_X_code_masks, shuffled_X_code_segment_ids, shuffled_Y = X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg_input_ids = shuffled_X_msg_input_ids[
                                     k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg_masks = shuffled_X_msg_masks[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg_segment_ids = shuffled_X_msg_segment_ids[
                                       k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]

        mini_batch_X_code_input_ids = shuffled_X_code_input_ids[
                                      k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_X_code_masks = shuffled_X_code_masks[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :,
                                  :]
        mini_batch_X_code_segment_ids = shuffled_X_code_segment_ids[
                                        k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]

        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (
        mini_batch_X_msg_input_ids, mini_batch_X_msg_masks, mini_batch_X_msg_segment_ids, mini_batch_X_code_input_ids,
        mini_batch_X_code_masks, mini_batch_X_code_segment_ids, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def mini_batches_updated(X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks,
                         X_code_segment_ids, Y, mini_batch_size=64, seed=0):
    ''' for training ; unbalanced data ; sample balanced data
    '''
    m = Y.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg_input_ids, shuffled_X_msg_masks, shuffled_X_msg_segment_ids, shuffled_X_code_input_ids, shuffled_X_code_masks, shuffled_X_code_segment_ids, shuffled_Y = X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))

        mini_batch_X_msg_input_ids = shuffled_X_msg_input_ids[indexes]
        mini_batch_X_msg_masks, mini_batch_X_msg_segment_ids = shuffled_X_msg_masks[indexes], \
                                                               shuffled_X_msg_segment_ids[indexes]
        mini_batch_X_code_input_ids, mini_batch_X_code_masks = shuffled_X_code_input_ids[indexes], \
                                                               shuffled_X_code_masks[indexes]
        mini_batch_X_code_segment_ids = shuffled_X_code_segment_ids[indexes]

        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (
        mini_batch_X_msg_input_ids, mini_batch_X_msg_masks, mini_batch_X_msg_segment_ids, mini_batch_X_code_input_ids,
        mini_batch_X_code_masks, mini_batch_X_code_segment_ids, mini_batch_Y)

        mini_batches.append(mini_batch)
    return mini_batches
def train_model(data, params):
    # preprocess on the code and msg data

    data_pad_msg, data_pad_code, data_labels,dict_msg,dict_code = data
    pad_msg_input_ids, pad_msg_input_masks, pad_msg_segment_ids = data_pad_msg
    pad_code_input_ids, pad_code_input_masks, pad_code_segment_ids = data_pad_code

    pad_msg_input_ids = np.array(pad_msg_input_ids)
    pad_msg_input_masks = np.array(pad_msg_input_masks)
    pad_msg_segment_ids = np.array(pad_msg_segment_ids)
    print("pad_msg_input_ids",pad_msg_input_ids.shape)
    print("pad_msg_input_masks",pad_msg_input_masks.shape)

    # pad the code changes data to num of files


    pad_code_input_ids = np.array(pad_code_input_ids)
    pad_code_input_masks = np.array(pad_code_input_masks)
    pad_code_segment_ids = np.array(pad_code_segment_ids)
    print("pad_code_input_ids",pad_code_input_ids.shape)
    print("pad_code_input_masks",pad_code_input_masks.shape)

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    # params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    # params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = BART4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    if params.load_model != None:
        model.load_state_dict(torch.load(params.load_model))

    criterion = nn.BCELoss()
    Adam_optimizer= torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    optimizer = Adam_optimizer


    # # logger = get_logger('log/CodeBERT/'+params.proj+".log")
    # starttime=time.time()
    # logger.info("training starting ")
    ## --------------- Training process ------------------ ##
    loss_res = []
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        step = 0
        # building batches for training model
        batches = mini_batches_updated(X_msg_input_ids=pad_msg_input_ids, X_msg_masks=pad_msg_input_masks,
                                       X_msg_segment_ids=pad_msg_segment_ids, X_code_input_ids=pad_code_input_ids,
                                       X_code_masks=pad_code_input_masks, X_code_segment_ids=pad_code_segment_ids,
                                       Y=data_labels, mini_batch_size=params.batch_size)
        for i, (batch) in enumerate(tqdm(batches)):
            step = step + 1
            msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = batch
            if torch.cuda.is_available():

                msg_input_id, msg_input_mask, code_input_id, code_input_mask, labels = torch.tensor(msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.tensor(code_input_id).cuda(), torch.tensor(
                    code_input_mask).cuda(), torch.cuda.FloatTensor(
                    labels.astype(int))
            else:
                print("-------------- Something Wrong with your GPU!!! ------------------")

                msg_input_id, msg_input_mask, code_input_id, code_input_mask, labels  = torch.tensor(msg_input_id).long(), torch.tensor(
                    msg_input_mask).long(),torch.tensor(code_input_id).long(),torch.tensor(code_input_mask).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()

            predict = model.forward(msg_input_id, msg_input_mask, code_input_id, code_input_mask,
                                    )
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch %i / %i  the step %i-- Total loss: %f' % (epoch, params.num_epochs, step, total_loss))
                # endtime=time.time()
                # dtime=endtime-starttime
                # logger.info('Epoch:[{}]\t loss={:.5f}\t time={:.3f}'.format(epoch, total_loss/150.0,dtime ))
                loss_res.append(total_loss.item())
                total_loss = 0
        save(model, params.save_dir, 'epoch', epoch, 'step', step)
    # logger.info("End training ")
    print("final loss : ", loss_res)
    write_csv(params.save_loss_path,file_name='loss.csv', data=loss_res)

def eval(labels, predicts, thresh=0.5):
    TP, FN, FP, TN = 0, 0, 0, 0
    for lable, predict in zip(labels, predicts):
        # print(predict)
        if predict >= thresh and lable == 1:
            TP += 1
        if predict >= thresh and lable == 0:
            FP += 1
        if predict < thresh and lable == 1:
            FN += 1
        if predict < thresh and lable == 0:
            TN += 1

    # print(TP)
    try:
        P = TP / (TP + FP)
        R = TP / (TP + FN)

        A = (TP + TN) / len(labels)
        E = FP / (TP + FP)

        # print(
        #     'Test data at Threshold %.2f -- Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f' % (
        #         thresh, A, E, P, R))
    except Exception:
        # division by zero
        pass
    return (A, E, P, R)

def evaluation_weight(data, params):
    # preprocess on the code and msg data
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    pad_code_input_ids, pad_code_input_masks, pad_code_segment_ids = pad_code
    pad_msg_input_ids, pad_msg_input_masks, pad_msg_segment_ids = pad_msg

    pad_msg_input_ids = np.array(pad_msg_input_ids)
    pad_msg_input_masks = np.array(pad_msg_input_masks)
    pad_msg_segment_ids = np.array(pad_msg_segment_ids)

    pad_code_input_ids = np.array(pad_code_input_ids)
    pad_code_input_masks = np.array(pad_code_input_masks)
    pad_code_segment_ids = np.array(pad_code_segment_ids)

    # build batches
    batches = mini_batches(X_msg_input_ids=pad_msg_input_ids, X_msg_masks=pad_msg_input_masks,
                           X_msg_segment_ids=pad_msg_segment_ids, X_code_input_ids=pad_code_input_ids,
                           X_code_masks=pad_code_input_masks, X_code_segment_ids=pad_code_segment_ids, Y=labels,
                           mini_batch_size=params.batch_size)
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = BART4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        msg_weight, cod_weight = [], []
        all_predict, all_label = list(), list()
        code_pred, msg_pred = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = batch

            sw_msg_input_id, sw_msg_input_mask, sw_msg_segment_id, sw_code_input_id, sw_code_input_mask, sw_code_segment_id = None, None, None, None, None, None

            if torch.cuda.is_available():
                msg_input_id, msg_input_mask, code_input_id, code_input_mask, labels = torch.tensor(
                    msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.tensor(code_input_id).cuda(), torch.tensor(
                    code_input_mask).cuda(),  torch.cuda.FloatTensor(
                    labels.astype(int))
                sw_msg_input_id, sw_msg_input_mask, sw_code_input_id, sw_code_input_mask = msg_input_id, torch.zeros_like(msg_input_mask), code_input_id, torch.zeros_like(code_input_mask)
                # print(type(sw_msg_input_id))
            else:
                print('cpu')

            predict = model.forward(msg_input_id, msg_input_mask, code_input_id, code_input_mask,
                                    )
            predict = predict.cpu().detach().numpy().tolist()
            all_predict += predict
            msg_predict = model.forward(msg_input_id, msg_input_mask, sw_code_input_id,
                                        sw_code_input_mask, )
            msg_predict = msg_predict.cpu().detach().numpy().tolist()
            msg_pred += msg_predict
            cod_predict = model.forward(sw_msg_input_id, sw_msg_input_mask, code_input_id,
                                        code_input_mask)
            cod_predict = cod_predict.cpu().detach().numpy().tolist()
            code_pred += cod_predict
            all_label += labels.tolist()

    print('Test data -- only left code predict: ')
    A, E, P, R = eval(all_label, code_pred, thresh=0.5)
    auc_score_code = roc_auc_score(y_true=all_label, y_score=code_pred)
    print(
        'Test data at Threshold 0.5 -- AUc: %.2f Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f' % (
            auc_score_code, A, E, P, R))

    print('Test data -- only left msg predict: ')

    A, E, P, R = eval(all_label, msg_pred, thresh=0.5)
    auc_score_msg = roc_auc_score(y_true=all_label, y_score=msg_pred)
    print(
        'Test data at Threshold 0.5 -- AUc: %.2f Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f' % (
            auc_score_msg, A, E, P, R))

def evaluation_model(data, params):
    # preprocess on the code and msg data
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    pad_code_input_ids, pad_code_input_masks, pad_code_segment_ids = pad_code
    pad_msg_input_ids, pad_msg_input_masks, pad_msg_segment_ids = pad_msg

    pad_msg_input_ids = np.array(pad_msg_input_ids)
    pad_msg_input_masks = np.array(pad_msg_input_masks)
    pad_msg_segment_ids = np.array(pad_msg_segment_ids)



    pad_code_input_ids = np.array(pad_code_input_ids)
    pad_code_input_masks = np.array(pad_code_input_masks)
    pad_code_segment_ids = np.array(pad_code_segment_ids)

    # build batches
    batches = mini_batches(X_msg_input_ids=pad_msg_input_ids, X_msg_masks=pad_msg_input_masks,
                           X_msg_segment_ids=pad_msg_segment_ids, X_code_input_ids=pad_code_input_ids,
                           X_code_masks=pad_code_input_masks, X_code_segment_ids=pad_code_segment_ids, Y=labels,
                           mini_batch_size=params.batch_size)

    # set up parameters

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = BART4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    if params.zero_shot:
        print("evaluating zero-shot model")
    else:
        model.load_state_dict(torch.load(params.load_model))

    ## ---------------------- Evalaution Process ---------------------------- ##
    model.eval()  # eval mode
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):

            msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = batch
            if torch.cuda.is_available():
                msg_input_id, msg_input_mask, code_input_id, code_input_mask, labels = torch.tensor(
                    msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.tensor(code_input_id).cuda(), torch.tensor(
                    code_input_mask).cuda(), torch.cuda.FloatTensor(
                    labels.astype(int))

            else:
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():

                predict = model.forward(msg_input_id, msg_input_mask, code_input_id, code_input_mask,
                                       )
                predict = predict.cpu().detach().numpy().tolist()
            else:

                predict = model.forward(msg_input_id, msg_input_mask, code_input_id, code_input_mask,
                                        )
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    # compute the AUC scores
    # write_csv(data=all_predict, file_name='predict.csv', path_dir=params.load_model.replace('.pt', '/'))
    A, E, P, R=eval(all_label, all_predict, thresh=0.5)
    auc_score = roc_auc_score(y_true=all_label, y_score=all_predict)
    # print('Test data -- AUC score:', auc_score)
    return (auc_score, A, E, P, R)


def one_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x / np.sum(x, axis=0)


if __name__ == '__main__':
    params = read_args().parse_args()
    if params.train is True:
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        data = pickle.load(open(params.train_data, 'rb'))

        ids, labels, msgs, codes = data
        data_len = len(ids)

        print(len(codes), len(ids))

        pad_msg = tokenization_for_codebert(data=msgs, max_length=params.msg_length, flag='msg', params=params)
        pad_code = tokenization_for_codebert(data=codes, max_length=params.code_length, flag='code', params=params)

        data = (pad_msg, pad_code, np.array(labels),dict_msg, dict_code)
        starttime = time.time()
        train_model(data=data, params=params)
        endtime = time.time()
        dtime = endtime - starttime
        print("程序运行时间：%.8s s" % dtime)  # 显示到微秒

    elif params.predict is True:
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        data = pickle.load(open(params.pred_data, 'rb'))

        ids, labels, msgs, codes = data
        data_len = len(ids)

        print(len(codes), len(ids))
        # tokenize the code and msg
        pad_msg = tokenization_for_codebert(data=msgs, max_length=params.msg_length, flag='msg', params=params)
        pad_code = tokenization_for_codebert(data=codes, max_length=params.code_length, flag='code', params=params)
        data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)


        starttime = time.time()
        if params.weight:
            evaluation_weight(data=data, params=params)
        else:
            auc_score, A, E, P, R = evaluation_model(data=data, params=params)
            print(
                'Test data at Threshold 0.5 -- AUc: %.2f Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f' % (
                    auc_score,
                    A, E, P, R))

        endtime = time.time()
        dtime = endtime - starttime
        print("程序运行时间：%.8s s" % dtime)  # 显示到微秒



        # # testing
        # if params.weight:
        #     evaluation_weight(data=data, params=params)
        # else:
        #     evaluation_model(data=data, params=params)
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()