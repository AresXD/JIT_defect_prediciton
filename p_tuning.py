path='../config_dataset/data/qt/cc2vec/qt_train.pkl'

import pickle
import os
import sys
import torch.nn as nn
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)





from transformers import get_linear_schedule_with_warmup, set_seed,AutoTokenizer,AutoModelForSequenceClassification
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
task = "mrpc"
peft_type = PeftType.LORA
device = "cuda"
num_epochs = 10
lr=1e-3

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
    parser.add_argument('-code_length', type=int, default=120, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=768, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-3, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=5, help='the number of epochs')
    parser.add_argument('-save-dir', type=str, default='p_tuning_roberta/test/qt', help='where to save the snapshot')


    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    return parser


##最长512
peft_config=PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


model_name_or_path = "microsoft/codebert-base"
tokenizer_name_or_path = "microsoft/codebert-base"

# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
# model=get_peft_model(model, peft_config)

def tokenize(path):
    data = read_pickle(path)
    ids, labels, msgs, codes = data

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,padding_side='right')
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # msg_codes= [msgs[i] + ' ' + codes[i] for i in range(len(msgs))]
    encoding = tokenizer(msgs, codes, truncation=True, max_length=492, padding='max_length')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return input_ids, attention_mask,labels

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



def mini_batches_train(input_ids, input_masks,Y, mini_batch_size=64,seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_input_ids, shuffled_X_masks, shuffled_Y = input_ids,input_masks, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))

        mini_batch_X_msg_input_ids = shuffled_X_input_ids[indexes]
        mini_batch_X_msg_masks = shuffled_X_masks[indexes]



        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (
        mini_batch_X_msg_input_ids, mini_batch_X_msg_masks, mini_batch_Y)

        mini_batches.append(mini_batch)
    return mini_batches


def train_model(data, params):
    # preprocess on the code and msg data

    pad_input_ids,pad_input_masks,data_labels = data
    pad_input_ids = np.array(pad_input_ids)
    pad_input_masks = np.array(pad_input_masks)
    data_labels = np.array(data_labels)




    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    # params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        model = model.cuda()
    if params.load_model != None:
        model.load_state_dict(torch.load(params.load_model))

    criterion = nn.BCELoss()
    Adam_optimizer= torch.optim.AdamW(model.parameters(), lr=params.l2_reg_lambda)
    optimizer = Adam_optimizer
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(pad_input_ids) * params.num_epochs),
        num_training_steps=(len(pad_input_ids) * params.num_epochs),
    )

    # optimizer = Adam_optimizer
    # if params.optimizer == 'Adam':
    #     optimizer = Adam_optimizer


    # # logger = get_logger('log/CodeBERT/'+params.proj+".log")
    # starttime=time.time()
    # logger.info("training starting ")
    ## --------------- Training process ------------------ ##
    loss_res = []
    # sigmoid = nn.Sigmoid()
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        step = 0
        # building batches for training model
        batches = mini_batches_train(input_ids=pad_input_ids, input_masks=pad_input_masks,Y=data_labels, mini_batch_size=batch_size)
        for i, (batch) in enumerate(tqdm(batches)):
            step = step + 1
            msg_input_id, msg_input_mask, labels = batch
            if torch.cuda.is_available():

                msg_input_id, msg_input_mask ,labels = torch.tensor(
                    msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.cuda.FloatTensor(
                    labels.astype(int))
            else:
                print("-------------- Something Wrong with your GPU!!! ------------------")

                # pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(
                #     pad_code).long(), torch.tensor(
                #     labels).float()

            optimizer.zero_grad()
            # print(msg_input_id.size(), msg_input_mask.size(), labels.size())
            predict=model(input_ids=msg_input_id, attention_mask=msg_input_mask)
            logits = predict.logits
            #将[16,2]logits转化为概率[16,]
            predict = torch.softmax(logits, dim=1)[:, 1]

            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if step % 100 == 0:
                print('Epoch %i / %i  the step %i-- Total loss: %f' % (epoch, params.num_epochs, step, total_loss))
                # endtime=time.time()
                # dtime=endtime-starttime
                # logger.info('Epoch:[{}]\t loss={:.5f}\t time={:.3f}'.format(epoch, total_loss/150.0,dtime ))
                loss_res.append(total_loss.item())
                total_loss = 0
        save(model, params.save_dir, 'epoch', epoch, 'step', step)
        model.save_pretrained(params.save_dir + '/epoch' + str(epoch))


    # logger.info("End training ")
    print("final loss : ", loss_res)
    # write_csv(params.save_loss_path,file_name='loss.csv', data=loss_res)

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
def mini_bacths_test(input_ids, input_masks, Y, mini_batch_size,seed=0):
    ''' for testing; put every data into it
        '''

    m = Y.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_input_ids, shuffled_X_input_masks, shuffled_Y =input_ids,input_masks, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1

    for k in range(0, num_complete_minibatches):
        mini_batch_X_input_ids = shuffled_X_input_ids[
                                     k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_input_masks = shuffled_X_input_masks[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]

        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_input_ids,mini_batch_X_input_masks, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



def evaluation_model(data, params):
    # preprocess on the code and msg data
    # preprocess on the code and msg data

    pad_input_ids, pad_input_masks, data_labels = data
    pad_input_ids = np.array(pad_input_ids)
    pad_input_masks = np.array(pad_input_masks)
    data_labels = np.array(data_labels)

    # build batches
    batches = mini_bacths_test(input_ids=pad_input_ids,input_masks= pad_input_masks, Y=data_labels, mini_batch_size=params.batch_size)

    # set up parameters



    # params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    # del params.no_cuda


    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    model = get_peft_model(model, peft_config)

    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    ## ---------------------- Evalaution Process ---------------------------- ##
    model.eval()  # eval mode
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):

            msg_input_id, msg_input_mask,labels = batch
            if torch.cuda.is_available():
                msg_input_id, msg_input_mask,labels = torch.tensor(
                    msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.cuda.FloatTensor(
                    labels.astype(int))

            else:
                print("GPU is not available")

            if torch.cuda.is_available():

                predict =model(input_ids=msg_input_id, attention_mask=msg_input_mask)
                logits = predict.logits
                # 将[16,2]logits转化为概率[16,]
                predict = torch.softmax(logits, dim=1)[:, 1]
                predict = predict.cpu().detach().numpy().tolist()

            else:

                print("GPU is not available")
            all_predict += predict
            all_label += labels.tolist()

    # compute the AUC scores
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
        path=params.train_data
        data=tokenize(path)
        starttime = time.time()
        train_model(data=data, params=params)
        endtime = time.time()
        dtime = endtime - starttime
        print("程序运行时间：%.8s s" % dtime)  # 显示到微秒






    elif params.predict is True:
        print("predicting lora-codebert model")

        path = params.pred_data
        data = tokenize(path)
        starttime = time.time()
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