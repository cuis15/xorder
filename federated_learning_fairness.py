import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import pickle
import sklearn.preprocessing as preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import random

class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        self.fc = nn.Linear(args.in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.fc.weight,0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.sigmoid(self.fc(x)).flatten()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default="adult", help="dataset to run(compas, framingham, adult, german)")
parser.add_argument('--path', type = str, default="data/original/adult_numerical-binsensitive.csv", help="dataset path")
parser.add_argument('--num_train_tasks', type = int, default=10, help="train tasks from 1 to 15")
parser.add_argument('--split', type = str, default="occupation", help="features for split the data")
parser.add_argument('--sensitive', type = str, default="race", help="(race, gender or other)")
parser.add_argument('--label', type = str, default="income", help="train tasks from 1 to 15")
parser.add_argument('--metric', type = str, default="xauc", help="metric of ranking fairness, xauc or prf")
parser.add_argument('--inner_lr', type = float, default=0.01, help="inner loop learning rate")
parser.add_argument('--outer_lr', type = float, default=0.0001, help="outer loop learning rate")
parser.add_argument('--s', type = int, default=10, help="support set size, (5,10,,,,)")
parser.add_argument('--q', type = int, default=10, help="query set size, (5,10,,,,)")
parser.add_argument('--epochs', type = int, default=1000, help="query set size, (5,10,,,,)")
parser.add_argument('--num_steps', type = tuple, default = (0, 1, 5, 10), help="test set training steps")
parser.add_argument('--test_s', type = int, default = 100, help="test set training steps")
parser.add_argument('--record_path', type = str, default = "record.txt", help="test set training steps")
parser.add_argument('--in_dim', type = int, default = 10, help="input dim")
args = parser.parse_known_args()[0]

compas = pd.read_csv('data/original/propublica-recidivism_numerical-binsensitive.csv')
adult = pd.read_csv('data/original/adult_numerical-binsensitive.csv')


def data_sample(data, args):

    num_train_tasks = args.num_train_tasks
    feat_split = args.split 
    feat_sensitive = args.sensitive
    label = args.label

    idx_tr = 0
    idx_te = 0
    a_y = data[[feat_sensitive, label]].copy(deep=True)

    
    data.drop(columns = label, inplace = False)
    data.drop(columns = feat_sensitive, inplace = False)
    data = pd.get_dummies(data)




    new_columns = [i for i in data.columns.tolist() if feat_split in i]

    trains = random.sample(new_columns, num_train_tasks)
    train_tasks = [[] for i in range(len(trains))]
    test_tasks = [[] for i in range(len(new_columns) - len(trains))]


    for idx, task_i in enumerate(new_columns):
        if task_i in trains:
            train_index = data[data[task_i]==True].index
            train_tasks[idx_tr] = train_index
            idx_tr+=1
        else:

            test_index = data[data[task_i]==True].index
            test_tasks[idx_te] = test_index
            idx_te+=1


    train_set = []
    test_set = []


    for idxs in train_tasks:
        train_label = a_y.loc[idxs,:][label].values.astype(np.float32)
        train_a = a_y.loc[idxs,:][feat_sensitive].values.astype(np.float32)
        train_feats = data.loc[idxs,:].values.astype(np.float32)
        train_set.append([train_feats, train_a, train_label])


    for idxs in test_tasks:
        test_labels = a_y.loc[idxs, :][label].values.astype(np.float32)
        test_a = a_y.loc[idxs,:][feat_sensitive].values.astype(np.float32)
        test_feats = data.loc[idxs,:].values.astype(np.float32)
        test_set.append([test_feats, test_a, test_labels])


    return train_set, test_set


def loss_fn(pred, label, attr, lamb):
    n = pred.shape[0]
    # Sample different example pairs
    I_1 = np.random.choice(n, n * 100)
    I_2 = np.random.choice(n, n * 100)
    A = (pred[I_1] - pred[I_2]) * (label[I_1] - label[I_2])
    B = (attr[I_1] - attr[I_2]) * (label[I_1] - label[I_2])
    loss_criterion = nn.BCELoss()
    loss_pred = loss_criterion(pred, label.float())
    if A.std().item() > 0.01:
        A_norm = (A - A.mean()) / A.std()
        B_norm = (B - B.mean()) / B.std()
        loss_corr = torch.abs((A_norm * B_norm).mean())
    else:
        loss_corr = 0.0
    loss = loss_pred + lamb * loss_corr

    return loss,loss_pred,loss_corr


def train(X,y,a,model,lr,max_epoch,lamb):
    epoch_cnt = 0
    loss_prev = 99999.0
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_model = deepcopy(model)
    for epoch in range(1, max_epoch):
        model.train()
        pred = model.forward(X)
        loss, _, _ = loss_fn(pred, y, a, lamb)
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model.forward(X)
        loss, _, _ = loss_fn(pred, y, a, lamb)
        if epoch < 1 or (loss.item() < loss_prev and abs(loss_prev - loss.item() / loss_prev) > 1e-6):
            best_model = deepcopy(model)
            epoch_cnt = 0
            loss_prev = loss.item()
        else:
            epoch_cnt += 1
        if epoch_cnt >= 5:
            break
    return best_model


def data_pre(arg):

    if arg.dataset=="adult":
        adult = pd.read_csv(args.path)   

        adult.dropna(how = "all", inplace = True)
        adult.drop(adult[adult["workclass"]=='?'].index,inplace=True)
        adult.drop(adult[adult["workclass"]=='Never-worked'].index, inplace=True)
        coun_sets = set(adult["workclass"])
        adult.tail()    

        occus_set = set(adult["occupation"])
        adult["income"] = (adult["income"]==">50K")
        adult["race"] = (adult["race"]=="White")
        coun_sets = set(adult["race"])  
    

        adult.drop(columns = "native-country", inplace=True)
        adult.reset_index(drop=True, inplace=True)
        return adult 

    else:
        pass
    

def data_fit(data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    return scaler

def data_transform(data, scalar):
    data = scalar.transform(data)
    return data


def eval_metric(pred, y, a, metric = 'xauc'):

    def xAUC_fast(a_score, b_score, a_label, b_label):
        a_num1 = np.sum(a_label)
        a_num0 = len(a_label) - a_num1
        b_num1 = np.sum(b_label)
        b_num0 = len(b_label) - b_num1  

        a_score1,a_score0 = a_score[a_label == 1],a_score[a_label == 0]
        b_score1,b_score0 = b_score[b_label == 1],b_score[b_label == 0] 

        ab_label = np.concatenate((np.ones(int(a_num1)),np.zeros(int(b_num0))))
        ab_score = np.concatenate((a_score1,b_score0))
        xauc_ab = roc_auc_score(ab_label,ab_score)  

        ba_label = np.concatenate((np.ones(int(b_num1)),np.zeros(int(a_num0))))
        ba_score = np.concatenate((b_score1,a_score0))
        xauc_ba = roc_auc_score(ba_label,ba_score)  

        return xauc_ab, xauc_ba, xauc_ab * a_num1 * b_num0 + xauc_ba * b_num1 * a_num0



    def cal_fairness_metric(pred, y, a, metric = "xauc"):
        a_idx, b_idx = np.where(a == 0), np.where(a == 1)
        a_score, b_score = pred[a_idx].reshape(-1), pred[b_idx].reshape(-1)
        a_label, b_label = y[a_idx].reshape(-1), y[b_idx].reshape(-1)
        if metric == "xauc":
            metric_ab, metric_ba, _ = xAUC_fast(a_score, b_score, a_label, b_label)
        else:
            metric_ab, metric_ba = pairwise_fast(a_score, b_score, a_label, b_label)
        return abs(metric_ab - metric_ba), metric_ab, metric_ba



    def pairwise_fast(a_score, b_score, a_label, b_label):
        a_num1 = np.sum(a_label)
        a_num0 = len(a_label) - a_num1
        b_num1 = np.sum(b_label)
        b_num0 = len(b_label) - b_num1  

        a_score1,a_score0 = a_score[a_label == 1],a_score[a_label == 0]
        b_score1,b_score0 = b_score[b_label == 1],b_score[b_label == 0] 

        ab_label = np.concatenate((np.ones(int(a_num1)),np.zeros(int(b_num0+a_num0))))
        ab_score = np.concatenate((a_score1,a_score0,b_score0))
        pair_ab = roc_auc_score(ab_label,ab_score) #[a=1, 0]    

        ba_label = np.concatenate((np.ones(int(b_num1)),np.zeros(int(a_num0+b_num0))))
        ba_score = np.concatenate((b_score1,b_score0, a_score0))
        pair_ba = roc_auc_score(ba_label,ba_score) #[b=1, 0]    
        return pair_ab, pair_ba 
    
    acc = np.sum(y==(pred>0.5))/len(y)
    auc = roc_auc_score(y, pred)
    
    if metric =='xauc':
        dis, a, b = cal_fairness_metric(pred, y, a, metric)
        return acc, auc, dis
    else:

        disab, disba = cal_fairness_metric(pre, y, a, metric)
        return acc, auc, disab-disba


    
def main(lamb_xorder):
    
    lr = 0.1
    max_epoch = 100
    num_run = 2
    
    data = data_pre(args)
    train_tasks, test_tasks = data_sample(data, args)
    
    train_x = np.concatenate(([t[0] for t in train_tasks]), axis = 0)
    train_y = np.concatenate(([t[2] for t in train_tasks]), axis = 0)
    train_a = np.concatenate(([t[1] for t in train_tasks]), axis = 0)

    scaler = data_fit(train_x)
    args.scaler = scaler
    args.in_dim = train_tasks[0][0].shape[1]

    train_x = data_transform(train_x, args.scaler)
    model = MODEL(args)

    var_X_train, var_y_train, var_a_train = Variable(torch.Tensor(train_x)), Variable(torch.Tensor(train_y)), Variable(torch.Tensor(train_a))   
    
    best_model = train(var_X_train, var_y_train, var_a_train, model, lr, max_epoch, lamb_xorder)
    test_x = np.concatenate(([t[0] for t in test_tasks]), axis = 0)
    test_y = np.concatenate(([t[2] for t in test_tasks]), axis = 0)
    test_a = np.concatenate(([t[1] for t in test_tasks]), axis = 0)
    test_idx = [i for i in range(test_y.shape[0])]
    random.shuffle(test_idx)
    
    result_split = []
    result_random = []
    
    idx_start = 0

    with torch.no_grad():
        best_model.eval()
     
        for task in test_tasks:
            test_x = data_transform(task[0], args.scaler)
            test_y = np.concatenate(([t[2] for t in train_tasks]), axis = 0)
            test_a = np.concatenate(([t[1] for t in train_tasks]), axis = 0)   
            var_X_test, var_y_test, var_a_test = Variable(torch.Tensor(test_x)), Variable(torch.Tensor(test_y)), Variable(torch.Tensor(test_a))   
            pred_test = best_model.forward(var_X_test).detach().numpy()
            
            acc, auc, dis = eval_metric(pred_test, test_y, test_a, metric = 'xauc')
            print(acc, auc, dis)
            result_split.append([acc, auc, dis])
            
            random_x = test_x[ test_idx[idx_start: idx_start + len(test_x)], :]
            random_x = data_transform(random_x, args.scaler)
            random_y = test_y[test_idx[idx_start: idx_start + len(test_x)]]
            random_a = test_a[test_idx[idx_start: idx_start + len(test_x)]]
            var_X_test, var_y_test, var_a_test = Variable(torch.Tensor(random_x)), Variable(torch.Tensor(random_y)), Variable(torch.Tensor(random_a))   
            pred_test = best_model.forward(var_X_test).detach().numpy()
            acc, auc, dis = eval_metric(pred_test, random_y, random_a, metric = 'xauc')
            print(acc, auc, dis)            
            result_random.append([acc, auc, dis])
        
    
    return np.mean(result_split, axis = 0), np.mean(result_random, axis = 0)


lambs_xorder = [0.02 * i for i in range(10)]
with open("record.txt", "a") as f:
    for lamb in lambs_xorder:
        split, random = main(lamb)
        f.write(str(lamb) + " split " + " ".join(split) + "\n")
        f.write(str(lamb) + " random " + " ".join(random) + "\n")

