import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import pickle
import sklearn.preprocessing as preprocessing
from rankboost import *
import numpy as np
from sklearn.model_selection import train_test_split

class MODEL(nn.Module):
    def __init__(self, dim):
        super(MODEL, self).__init__()
        self.fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.fc.weight,0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.sigmoid(self.fc(x)).flatten()

# Loss function for corr-reg
def loss_fn(pred,label,attr,lamb):
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


def train_eval_model(model, X_train, y_train, X_test):
    model.fit(X_train,y_train)
    pred_train = model.predict_proba(X_train)[:,1]
    pred_test = model.predict_proba(X_test)[:,1]
    return pred_train,pred_test


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


def determine_stop(disparity_train):
    min_disparity = 1.0
    min_idx = 0
    stop_cnt = 0
    for i in range(len(disparity_train)):
        if disparity_train[i] < 0.01:
            return i
        if disparity_train[i] < min_disparity:
            min_disparity = disparity_train[i]
            min_idx = i
            stop_cnt = 0
        else:
            if disparity_train[i] > disparity_train[i - 1]:
                stop_cnt += 1

        if stop_cnt >= 2:
            return min_idx
    return min_idx


def run_experiment(dataset,eval_metric,classifier):
    lr = 0.1
    max_epoch = 100
    num_run = 2

    # Set up weights for corr-reg, ensuring that the stop criteria take effect before the upper limits are reached
    if classifier == "lr":
        if dataset == "compas":
            lambs_corr = [0.03 * i for i in range(10)]
        elif dataset == "framingham":
            lambs_corr = [0.01 * i for i in range(10)]
        elif dataset == "adult":
            lambs_corr = [0.02 * i for i in range(10)]
        elif dataset == "german":
            lambs_corr = [0.02 * i for i in range(10)]
        else:
            lambs_corr = [0.02 * i for i in range(10)]

        aucs_corr = np.zeros((len(lambs_corr), num_run))
        disparity_train_corr = np.zeros((len(lambs_corr), num_run))
        disparity_test_corr = np.zeros((len(lambs_corr), num_run))

    aucs_un = np.zeros(num_run)
    disparity_train_un = np.zeros(num_run)
    disparity_test_un = np.zeros(num_run)

    aucs_log = np.zeros(num_run)
    disparity_train_log = np.zeros(num_run)
    disparity_test_log = np.zeros(num_run)

    # Set up weights for xorder, ensuring that the stop criteria take effect before the upper limits are reached
    if dataset == "compas":
        lambs_xorder = [0.02 * i for i in range(10)]
    elif dataset == "framingham":
        lambs_xorder = [0.02 * i for i in range(15)]
    elif dataset == "german":
        lambs_xorder = [0.02 * i for i in range(10)]
    elif dataset == "adult":
        lambs_xorder = [0.02 * i for i in range(10)]
    else:
        lambs_xorder = [0.02 * i for i in range(10)]
    aucs_xorder = np.zeros((len(lambs_xorder), num_run))
    disparity_train_xorder = np.zeros((len(lambs_xorder), num_run))
    disparity_test_xorder = np.zeros((len(lambs_xorder), num_run))

    for run_idx in range(num_run):
        print("Experiment index: {}/{}".format(run_idx + 1,num_run))

        fin = open("data/preprocessed/" + dataset + "_data" + '.pkl', 'rb')
        data_dict = pickle.load(fin)
        X, y, a = data_dict["X"], data_dict["y"].astype(np.int), data_dict["a"].astype(np.float32)


#         fin = open("data/preprocessed/" + dataset + "_split_idx_" + str(run_idx) + '.pkl', 'rb')
#         data_dict = pickle.load(fin)
#         idx_train, idx_test = data_dict["idx_train"], data_dict["idx_test"]

        all_idx = np.array(range(len(y)))
        idx_train, idx_test = train_test_split(all_idx, train_size = 0.7)

        X_train, y_train, a_train = X[idx_train,:], y[idx_train].astype(np.int), a[idx_train].astype(np.float32)
        X_test, y_test, a_test = X[idx_test, :], y[idx_test].astype(np.int), a[idx_test].astype(np.float32)

        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        if classifier == "lr":
            print("Running unadjusted and corr-reg...")
            # Train corr-reg with different weights of regularization
            var_X_train, var_y_train, var_a_train = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train)), Variable(torch.Tensor(a_train))
            var_X_test, var_y_test, var_a_test = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(y_test)), Variable(torch.Tensor(a_test))
            for (lamb_idx, lamb) in enumerate(lambs_corr):
                model = MODEL(dim=X_train.shape[1])
                best_model = train(var_X_train,var_y_train,var_a_train,model,lr,max_epoch,lamb)

                with torch.no_grad():
                    best_model.eval()
                    pred_train = best_model.forward(var_X_train)
                    pred_train = pred_train.detach().numpy()
                    pred_test = best_model.forward(var_X_test).detach().numpy()
                    # The case of weight = 0 is equivalent to unadjusted
                    if lamb < 1e-6:
                        pred_train_un = pred_train
                        pred_test_un = pred_test

                    auc_test = roc_auc_score(y_test, pred_test)
                    aucs_corr[lamb_idx,run_idx] = auc_test

                    disparity_train,_ ,_ = cal_fairness_metric(pred_train,y_train,a_train,metric=eval_metric)
                    disparity_train_corr[lamb_idx,run_idx] = disparity_train

                    disparity_test,_ ,_ = cal_fairness_metric(pred_test,y_test,a_test,metric=eval_metric)
                    disparity_test_corr[lamb_idx, run_idx] = disparity_test
        else:
            print("Running unadjusted...")
            model_rb = BipartiteRankBoost(n_estimators=50, verbose=1, learning_rate=1.0)
            pred_train_un, pred_test_un = train_eval_model(model_rb, X_train, y_train, X_test)

        auc_test = roc_auc_score(y_test, pred_test_un)
        aucs_un[run_idx] = auc_test
        disparity_train, dis_ab, dis_ba = cal_fairness_metric(pred_train_un, y_train, a_train, metric=eval_metric)
        disparity_train_un[run_idx] = dis_ab - dis_ba
        disparity_test, dis_ab, dis_ba = cal_fairness_metric(pred_test_un, y_test, a_test, metric=eval_metric)
        disparity_test_un[run_idx] = dis_ab - dis_ba

        # Rescale prediction scores to mitigate extreme distribution in german dataset(Otherwise both methods of post-logit and xorder will fail to address unfairness)
        
######################################################
        # if dataset == "german" and classifier == "rb":
        #     pred_test_un = (pred_test_un - pred_test_un.mean()) / pred_test_un.std()
        #     pred_test_un = pred_test_un * pred_train_un.std() + pred_train_un.mean()
######################################################


        print("Running post-log...")
        # Sorting the instances of group a and b
        tr_a_score_sort, tr_b_score_sort, tr_a_label_sort, tr_b_label_sort = generate_sorted_groups(pred_train_un, y_train,
                                                                                                    a_train)
        te_a_score_sort, te_b_score_sort, te_a_label_sort, te_b_label_sort = generate_sorted_groups(pred_test_un, y_test,
                                                                                                    a_test)
        beta = -2.0
        paras, disparities_train = [], []
        # Searching on the space of \alpha with fixed \beta, this is the same as in the supplemental material of post-logit
        for a_idx in range(100):
            alpha = 0.1 * a_idx
            adjust_tr_b_score_sort = 1 / (1 + np.exp(-(alpha * tr_b_score_sort + beta)))
            disparity_train, _, _ = cal_fairness_metric_by_groups(tr_a_score_sort, adjust_tr_b_score_sort, tr_a_label_sort,
                                                           tr_b_label_sort, eval_metric)
            paras.append(alpha)
            disparities_train.append(disparity_train)

        paras = np.array(paras)
        disparities_train = np.array(disparities_train)

        # Find the optimal \alpha to achieve fair result on training data
        opt_idx = disparities_train.argsort()[0]
        opt_para = paras[opt_idx]

        adjust_tr_b_score_sort = 1 / (1 + np.exp(-(opt_para * tr_b_score_sort + beta)))
        disparity_train, _, _ = cal_fairness_metric_by_groups(tr_a_score_sort, adjust_tr_b_score_sort,
                                                                   tr_a_label_sort, tr_b_label_sort, eval_metric)

        adjust_te_b_score_sort = 1 / (1 + np.exp(-(opt_para * te_b_score_sort + beta)))
        disparity_test, _, _ = cal_fairness_metric_by_groups(te_a_score_sort, adjust_te_b_score_sort,
                                                                 te_a_label_sort, te_b_label_sort, eval_metric)
        auc_test = roc_auc_score(np.concatenate((te_a_label_sort, te_b_label_sort)),
                                     np.concatenate((te_a_score_sort, adjust_te_b_score_sort)))

        aucs_log[run_idx] = auc_test
        disparity_train_log[run_idx] = disparity_train
        disparity_test_log[run_idx] = disparity_test

        print("Running xorder...")
        k = y_train.sum() * (1 - y_train).sum()
        for (lamb_idx,lamb) in enumerate(lambs_xorder):
            post_tr_b_score, _ = post_b_score(tr_a_score_sort, tr_b_score_sort,
                                                      np.concatenate(([0], tr_a_label_sort), axis=0),
                                                      np.concatenate(([0], tr_b_label_sort), axis=0), lamb * k, _type=eval_metric)
            post_te_b_score = post_score(tr_b_score_sort, post_tr_b_score, te_b_score_sort)

            post_auc = roc_auc_score(list(te_a_label_sort) + list(te_b_label_sort),
                                          list(te_a_score_sort) + list(post_te_b_score))

            _, m_ab_tr, m_ba_tr = cal_fairness_metric_by_groups(tr_a_score_sort, post_tr_b_score, tr_a_label_sort,
                                                           tr_b_label_sort, eval_metric)
            _, m_ab_te, m_ba_te = cal_fairness_metric_by_groups(te_a_score_sort, post_te_b_score, te_a_label_sort, te_b_label_sort, eval_metric)
            disparity_train_xorder[lamb_idx,run_idx] = m_ab_tr - m_ba_tr
            disparity_test_xorder[lamb_idx,run_idx] = m_ab_te - m_ba_te
            aucs_xorder[lamb_idx,run_idx] = post_auc
    print("Result of unadjusted:")
    print("Train disparity: ", disparity_train_un)
    print("Test disparity: ", disparity_test_un)
    print("Test total AUC: {:.3f}".format(aucs_un.mean()))
    if classifier == "lr":
        print("Result for corr-reg under different weights:")
        print("Train disparity:", array2str(disparity_train_corr.mean(1)))
        print("Test disparity: ", array2str(disparity_test_corr.mean(1)))
        print("Test total AUC: ",array2str(aucs_corr.mean(1)))
        stop_idx = determine_stop(disparity_train_corr.mean(1))
        print(stop_idx)
        print("After determining stopping weights:")
        print("Train disparity:", array2str(disparity_train_corr.mean(1)[:stop_idx + 1]))
        print("Test disparity: ", array2str(disparity_test_corr.mean(1)[:stop_idx + 1]))
        print("Test total AUC: ",array2str(aucs_corr.mean(1)[:stop_idx + 1]))
    print("Result for post-log:")
    print("Train disparity:{:.3f}".format(disparity_train_log.mean()))
    print("Test disparity: {:.3f}".format(disparity_test_log.mean()))
    print("Test total AUC: {:.3f}".format(aucs_log.mean()))
    print("Result for xorder under different weights:")
    print("Train disparity:  ", disparity_train_xorder)
    print("Test disparity: ",  disparity_test_xorder)
    print("Test total AUC: ",array2str(aucs_xorder.mean(1)))
    stop_idx = determine_stop(disparity_train_xorder.mean(1))
    print(stop_idx)
    print("After determining stopping weights:")
    print("Train disparity:", array2str(disparity_train_xorder.mean(1)[:stop_idx + 1]))
    print("Test disparity: ", array2str(disparity_test_xorder.mean(1)[:stop_idx + 1]))
    print("Test total AUC: ", array2str(aucs_xorder.mean(1)[:stop_idx + 1]))