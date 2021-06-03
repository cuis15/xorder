from utils import cal_fairness_metric
import pickle
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from rankboost import *
from sklearn.linear_model import LogisticRegression


def train_eval_model(model, X_train, y_train, X_test):
    model.fit(X_train,y_train)
    pred_train = model.predict_proba(X_train)[:,1]
    pred_test = model.predict_proba(X_test)[:,1]
    return pred_train,pred_test


def verify_split(pred_train,y_train,a_train,pred_test,y_test,a_test):
    delta_xAUC_tr, xauc_ab_tr, xauc_ba_tr = cal_fairness_metric(pred_train, y_train, a_train, metric="xauc")

    delta_PRF_tr, prf_ab_tr, prf_ba_tr = cal_fairness_metric(pred_train, y_train, a_train, metric="prf")


    delta_xAUC, xauc_ab_te, xauc_ba_te = cal_fairness_metric(pred_test, y_test, a_test, metric="xauc")


    delta_PRF, prf_ab_te, prf_ba_te = cal_fairness_metric(pred_test, y_test, a_test, metric="prf")


    return (xauc_ab_tr - xauc_ba_tr) * (xauc_ab_te - xauc_ba_te) >0 , (prf_ab_tr - prf_ba_tr) * (prf_ab_te - prf_ba_te) >0
    



np.random.seed()
for dataset in ["german"]:
    fin = open("preprocessed/" + dataset + "_data" + '.pkl', 'rb')
    data_dict = pickle.load(fin)
    X, y, a = data_dict["X"], data_dict["y"].astype(np.int), data_dict["a"].astype(np.float32)

    all_idx = np.array(range(len(y)))
    num_=0
    for i in range(50):
        idx_train, idx_test = train_test_split(all_idx, train_size=0.7)
        X_train, y_train, a_train = X[idx_train,:], y[idx_train].astype(np.int), a[idx_train].astype(np.float32)
        X_test, y_test, a_test = X[idx_test, :], y[idx_test].astype(np.int), a[idx_test].astype(np.float32) 

        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)   

        model_rb = BipartiteRankBoost(n_estimators=50, verbose=1, learning_rate=1.0)
        pred_train, pred_test = train_eval_model(model_rb, X_train, y_train, X_test)
        del_axuc, del_prf = verify_split(pred_train,y_train,a_train,pred_test,y_test,a_test)

        if del_axuc == True and del_prf == True:
            num_+=1
            print(1)
        else:
            print(0)
        i+=1
            
print(num_)

