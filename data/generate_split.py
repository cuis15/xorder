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
    print("xAUC", delta_xAUC_tr)
    delta_PRF_tr, prf_ab_tr, prf_ba_tr = cal_fairness_metric(pred_train, y_train, a_train, metric="prf")
    print("PRF", delta_PRF_tr)

    delta_xAUC, xauc_ab_te, xauc_ba_te = cal_fairness_metric(pred_test, y_test, a_test, metric="xauc")
    print("xAUC", delta_xAUC)
    delta_PRF, prf_ab_te, prf_ba_te = cal_fairness_metric(pred_test, y_test, a_test, metric="prf")
    print("PRF", delta_PRF)

    print(xauc_ab_tr - xauc_ba_tr,xauc_ab_te - xauc_ba_te,prf_ab_tr - prf_ba_tr,prf_ab_te - prf_ba_te)
    if abs(delta_xAUC_tr - delta_xAUC) < 0.2 and abs(delta_PRF_tr - delta_PRF) < 0.2:
        if delta_xAUC_tr > 0.01 and delta_xAUC > 0.01 and delta_PRF_tr > 0.01 and delta_PRF > 0.01:
            if (xauc_ab_tr - xauc_ba_tr) * (xauc_ab_te - xauc_ba_te) > 0 and (prf_ab_tr - prf_ba_tr) * (prf_ab_te - prf_ba_te) > 0:
                print(xauc_ab_tr - xauc_ba_tr, xauc_ab_te - xauc_ba_te, prf_ab_tr - prf_ba_tr, prf_ab_te - prf_ba_te)
                return True
            else:
                return False
        else:
            return True
    else:
        return False




np.random.seed(12345)
for dataset in ["german"]:
    fin = open("preprocessed/" + dataset + "_data" + '.pkl', 'rb')
    data_dict = pickle.load(fin)
    X, y, a = data_dict["X"], data_dict["y"].astype(np.int), data_dict["a"].astype(np.float32)
    for idx in range(10):
        # Generate train-test split without inconsistent results(for example xAUC(a,b) > xAUC(b,a) in training data but xAUC(a,b) < xAUC(b,a) in test data)
        while True:
            all_idx = np.array(range(len(y)))
            idx_train, idx_test = train_test_split(all_idx, train_size=0.7)
            X_train, y_train, a_train = X[idx_train,:], y[idx_train].astype(np.int), a[idx_train].astype(np.float32)
            X_test, y_test, a_test = X[idx_test, :], y[idx_test].astype(np.int), a[idx_test].astype(np.float32)

            scaler = preprocessing.StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            model_rb = BipartiteRankBoost(n_estimators=50, verbose=1, learning_rate=1.0)
            pred_train, pred_test = train_eval_model(model_rb, X_train, y_train, X_test)
            verify1 = verify_split(pred_train,y_train,a_train,pred_test,y_test,a_test)

            model_lr = LogisticRegression()
            pred_train, pred_test = train_eval_model(model_lr, X_train, y_train, X_test)
            verify2 = verify_split(pred_train, y_train, a_train, pred_test, y_test, a_test)

            if verify1 == True and verify2 == True:
                break

        idx_dict = {"idx_train":idx_train,"idx_test":idx_test}
        fout = open("preprocessed/" + dataset + "_split_idx_" + str(idx) + '.pkl', 'wb')
        pickle.dump(idx_dict,fout)

