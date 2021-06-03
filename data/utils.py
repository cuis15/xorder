import numpy as np
from sklearn.metrics import roc_auc_score
from numba import jit


def array2str(tmp_array, sep = " "):
    str_list = ["{:.3f}".format(tmp_item) for tmp_item in tmp_array]
    return sep.join(str_list)


def generate_sorted_groups(pred, y, a):
    a_idx = np.where(a == 0)
    b_idx = np.where(a == 1)
    b_score = pred[b_idx].reshape(-1)
    b_index = np.argsort(-b_score)
    b_score_sort = b_score[b_index]
    b_label = y[b_idx]
    b_label_sort = b_label[b_index]

    a_score = pred[a_idx].reshape(-1)
    a_index = np.argsort(-a_score)
    a_score_sort = a_score[a_index]
    a_label = y[a_idx]
    a_label_sort = a_label[a_index]

    return a_score_sort,b_score_sort,a_label_sort,b_label_sort


def cal_fairness_metric_by_groups(a_score, b_score, a_label, b_label, metric = "xauc"):
    if metric == "xauc":
        metric_ab, metric_ba, _ = xAUC_fast(a_score, b_score, a_label, b_label)
    else:
        metric_ab, metric_ba = pairwise_fast(a_score, b_score, a_label, b_label)
    return abs(metric_ab - metric_ba),metric_ab,metric_ba


def cal_fairness_metric(pred, y, a, metric = "xauc"):
    a_idx, b_idx = np.where(a == 0), np.where(a == 1)
    a_score, b_score = pred[a_idx].reshape(-1), pred[b_idx].reshape(-1)
    a_label, b_label = y[a_idx].reshape(-1), y[b_idx].reshape(-1)
    if metric == "xauc":
        metric_ab, metric_ba, _ = xAUC_fast(a_score, b_score, a_label, b_label)
    else:
        metric_ab, metric_ba = pairwise_fast(a_score, b_score, a_label, b_label)
    return abs(metric_ab - metric_ba),metric_ab,metric_ba


def AUC(score, label):
    ###[from big to small]
    sum_ = 0
    num = len(label)
    for i in range(num):
        for j in range(num):
            if label[i]==1 and label[j]==0:
                if score[i]>score[j]:    
                    sum_ += 1

    return sum_/(np.sum(label)*(num-np.sum(label))), sum_


def xAUC(a_score, b_score, a_label, b_label):
    sum_ab = 0
    sum_ba = 0
    numa = len(a_label)
    numb = len(b_label)
    a_num1 = np.sum(a_label)
    a_num0 = len(a_label) - a_num1
    b_num1 = np.sum(b_label)
    b_num0 = len(b_label) - b_num1
    for i in range(numa):
        for j in range(numb):
            if a_label[i] ==1 and b_label[j] ==0:
                if a_score[i]>b_score[j]:
                    sum_ab+=1
            elif a_label[i]==0 and b_label[j]==1:
                if b_score[j]>a_score[i]:
                    sum_ba+=1
    return sum_ab/(a_num1*b_num0), sum_ba/(b_num1*a_num0), sum_ab+sum_ba 


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


def post_score(train_score, train_score_post, test_score):
    tep_id = 0
    bins = [[] for i in range(len(train_score)+1)]
    for i in range(len(test_score)):
        s = test_score[i]
        if s>train_score[0]:
            bins[0].append(s)
        elif s<=train_score[-1]:
            bins[-1].append(s)
        else:
            for j in range(tep_id,len(train_score)):
                if train_score[j-1]>=s and train_score[j]<s:
                    bins[j].append(s)
                    tep_id = j
                    break
    changed_b_score  = []
    for bin_ in range(len(bins)):
        for item in range(len(bins[bin_])):
            num = (len(bins[bin_]))
            if bin_==0:
                changed_b_score.append((item)*train_score_post[bin_]/num+(num-item)/num)
            elif bin_==len(train_score_post):
                changed_b_score.append((num -item)*train_score_post[bin_-1]/num)
            else:
                changed_b_score.append((item)*train_score_post[bin_]/num + (num-item)*train_score_post[bin_-1]/num)
        
    return np.array(changed_b_score)


@jit(nopython=True)
def maxAUC(a_label, b_label):

    M = len(a_label)-1
    N = len(b_label)-1
    a_1 = np.sum(a_label)
    b_1 = np.sum(b_label)
    path = np.zeros((M+1, N+1,2,2))

    cost = np.zeros((M+1, N+1))
    for i in range(1,M+1):
        if a_label[i]==1:
            cost[i,0] = N-b_1 + cost[i-1, 0]
        else:
            cost[i,0] = cost[i-1,0]
        path[i,0,:,:] = np.array([[i-1, 0], [ i, 0]])

    for i in range(1,N+1):
        if b_label[i]==1:
            cost[0, i] = cost[0,i-1]+ M - a_1
        else:
            cost[0, i] = cost[0,i-1]
        path[0,i,:,:] = np.array([[0, i-1],[0, i]])


    for i in range(2, M+1+N+1):
        for j in range(max(1, i-N), min(i, M+1)): # j[1, i-1]

            if i-j+1>N or a_label[j]==0:
                tep_b = 0 
            else:
                tep_b = N - (i-j) - np.sum(b_label[i-j+1:])

            if j+1>M or b_label[i-j]==0:
                tep_a = 0
            else:
                tep_a = M - j -np.sum(a_label[j+1:])

            if cost[j-1, i-j] + tep_b > cost[j, i-j-1] + tep_a:
                cost[j, i-j] = cost[j-1, i-j] + tep_b
                path[j, i-j,:,:] = np.array([[j-1, i-j], [j, i-j]])

            else:
                cost[j, i-j] = cost[j, i-j-1] + tep_a
                path[j, i-j,:,:] = np.array([[j, i-j-1], [j, i-j]])
    return cost[M,N], path


@jit(nopython=True)
def xAUC_post(a_label, b_label, lamb):
    M = len(a_label)-1
    N = len(b_label)-1
    a_1 = np.sum(a_label)
    b_1 = np.sum(b_label)

    a_1_b_0 = a_1*(N-b_1)
    b_1_a_0 = b_1*(M - a_1)

    path = np.zeros((M+1, N+1,2,2))
    cost_unfair = np.zeros((M+1, N+1))
    cost = np.zeros((M+1, N+1))
    for i in range(1,M+1):
        if a_label[i]==1:
            cost_unfair[i, 0] = (N-b_1)/a_1_b_0*lamb + cost_unfair[i-1,0]
            cost[i,0] = N-b_1 + cost[i-1, 0] 
        else:
            cost_unfair[i, 0] = cost_unfair[i-1,0]
            cost[i,0] = cost[i-1,0]
        path[i,0,:,:] = np.array([[i-1, 0], [ i, 0]])

    for i in range(1,N+1):
        if b_label[i]==1:
            cost_unfair[0,i] = -(M-a_1)/b_1_a_0*lamb + cost_unfair[0, i-1]
            cost[0, i] = cost[0,i-1] + M - a_1
        else:
            cost[0, i] = cost[0,i-1]
            cost_unfair[0, i] = cost_unfair[0,i-1]
        path[0,i,:,:] = np.array([[0, i-1],[0, i]])

    for i in range(2, M+1+N+1):
        for j in range(max(1, i-N), min(i, M+1)): # j[1, i-1]

            if i-j+1>N or a_label[j]==0:
                tep_b = 0 
                tep_unfair_b = 0
            else:
                tep_b = N - (i-j) - np.sum(b_label[i-j+1:])
                tep_unfair_b = tep_b/a_1_b_0*lamb 

            if j+1>M or b_label[i-j]==0:
                tep_a = 0
                tep_unfair_a = 0
            else:
                tep_a = M - j -np.sum(a_label[j+1:])
                tep_unfair_a = -tep_a/b_1_a_0*lamb

            if cost[j-1, i-j] + tep_b - abs(tep_unfair_b + cost_unfair[j-1, i-j]) > cost[j, i-j-1] + tep_a - abs(tep_unfair_a + cost_unfair[j, i-j-1]):
                cost_unfair[j, i-j] = tep_unfair_b + cost_unfair[j-1, i-j]
                cost[j, i-j] = cost[j-1, i-j] + tep_b 
                path[j, i-j,:,:] = np.array([[j-1, i-j], [j, i-j]])

            else:
                cost_unfair[j, i-j] = tep_unfair_a + cost_unfair[j, i-j-1]
                cost[j, i-j] = cost[j, i-j-1] + tep_a 
                path[j, i-j,:,:] = np.array([[j, i-j-1], [j, i-j]])

    return cost, path, cost_unfair

@jit(nopython=True)
def xAUC_post_(a_label, b_label, lamb):
    M = len(a_label)-1
    N = len(b_label)-1
    a_1 = np.sum(a_label)
    b_1 = np.sum(b_label)

    a_1_b_0 = a_1*(N-b_1)
    b_1_a_0 = b_1*(M - a_1)

    path = np.zeros((M+1, N+1,2,2))
    cost_unfair = np.zeros((M+1, N+1))
    cost = np.zeros((M+1, N+1))
    for i in range(1,M+1):
        if a_label[i]==1:
            cost_unfair[i, 0] = (N-b_1)/a_1_b_0 * lamb + cost_unfair[i-1,0]
            cost[i,0] = N-b_1 + cost[i-1, 0] 
        else:
            cost_unfair[i, 0] = cost_unfair[i-1,0]
            cost[i,0] = cost[i-1,0]
        path[i,0,:,:] = np.array([[i-1, 0], [ i, 0]])

    for i in range(1,N+1):
        if b_label[i]==1:
            cost_unfair[0,i] = -(M - a_1) / b_1_a_0 * lamb + cost_unfair[0, i-1]
            cost[0, i] = cost[0,i-1] + M - a_1
        else:
            cost[0, i] = cost[0,i-1]
            cost_unfair[0, i] = cost_unfair[0,i-1]
        path[0,i,:,:] = np.array([[0, i-1],[0, i]])

    for i in range(2, M+1+N+1):
        # print(i)
        for j in range(max(1, i-N), min(i, M+1)): # j[1, i-1]

            if  a_label[j]==0:
                tep_b = 0 
                tep_unfair_b = 0
            else:
                tep_b = N - (i-j) - np.sum(b_label[i-j+1:])
                tep_unfair_b = tep_b/a_1_b_0*lamb 

            if b_label[i-j]==0:
                tep_a = 0
                tep_unfair_a = 0
            else:
                tep_a = M - j -np.sum(a_label[j+1:])
                tep_unfair_a = -tep_a/b_1_a_0*lamb

            if cost[j-1, i-j] + tep_b - abs(tep_unfair_b + cost_unfair[j-1, i-j]) > cost[j, i-j-1] + tep_a - abs(tep_unfair_a + cost_unfair[j, i-j-1]):
                cost_unfair[j, i-j] = tep_unfair_b + cost_unfair[j-1, i-j]
                cost[j, i-j] = cost[j-1, i-j] + tep_b 
                path[j, i-j,:,:] = np.array([[j-1, i-j], [j, i-j]])

            else:
                cost_unfair[j, i-j] = tep_unfair_a + cost_unfair[j, i-j-1]
                cost[j, i-j] = cost[j, i-j-1] + tep_a 
                path[j, i-j,:,:] = np.array([[j, i-j-1], [j, i-j]])

    return cost, path, cost_unfair


@jit(nopython=True)
def pairwise_post(a_label, b_label, lamb):
###a, b has been sorted decreasing sort.
    M = len(a_label)-1
    N = len(b_label)-1
    a_1 = np.sum(a_label)
    b_1 = np.sum(b_label)

    a_1_0 = a_1*((N-b_1)+(M - a_1))
    b_1_0 = b_1*((M - a_1)+(N-b_1))

    path = np.zeros((M+1, N+1,2,2))
    cost_unfair = np.zeros((M+1, N+1))
    cost = np.zeros((M+1, N+1))

    zeros_mat = np.zeros((M+1, N+1))
    zeros_mat[0,0] = ((N-b_1)+(M - a_1))

    for i in range(1,N+1):
        if b_label[i]==1:
            zeros_mat[0,i] = zeros_mat[0,i-1]
        else:
            zeros_mat[0,i] = zeros_mat[0,i-1]-1 

    for i in range(1,M+1):
        if a_label[i]==0:
            zeros_mat[i,0] = zeros_mat[i-1,0]-1
        else:
            zeros_mat[i,0] = zeros_mat[i-1,0]
        for j in range(1,N+1):
            if b_label[j]==0:
                zeros_mat[i,j] = zeros_mat[i,j-1]-1
            else:
                zeros_mat[i,j] = zeros_mat[i,j-1]
    for i in range(1,M+1):
        if a_label[i]==1:
            cost_unfair[i, 0] = zeros_mat[i,0]/a_1_0*lamb + cost_unfair[i-1,0]
            cost[i,0] = N-b_1 + cost[i-1, 0] 
        else:
            cost_unfair[i, 0] =  cost_unfair[i-1,0]
            cost[i,0] = cost[i-1,0]
        path[i,0,:,:] = np.array([[i-1, 0], [ i, 0]])

    for i in range(1,N+1):
        if b_label[i]==1:
            cost_unfair[0,i] = -zeros_mat[0,i]/b_1_0*lamb + cost_unfair[0, i-1]
            cost[0, i] = cost[0,i-1] + M - a_1
        else:

            cost[0, i] = cost[0,i-1]
            cost_unfair[0, i] = cost_unfair[0, i-1]
        path[0,i,:,:] = np.array([[0, i-1],[0, i]])

    for i in range(2, M+1+N+1):
        for j in range(max(1, i-N), min(i, M+1)): # j[1, i-1]

            if  a_label[j]==0:
                tep_b = 0 
                tep_unfair_b = 0
            else:
                tep_b = N - (i-j) - np.sum(b_label[i-j+1:])
                tep_unfair_b = zeros_mat[j,i-j]/a_1_0*lamb 


            if  b_label[i-j]==0:
                tep_a = 0
                tep_unfair_a = 0
            else: 
                tep_a = M - j -np.sum(a_label[j+1:])
                tep_unfair_a = -zeros_mat[j,i-j]/b_1_0*lamb

            if cost[j-1, i-j] + tep_b - abs(tep_unfair_b + cost_unfair[j-1, i-j]) > cost[j, i-j-1] + tep_a - abs(tep_unfair_a + cost_unfair[j, i-j-1]):

                cost_unfair[j, i-j] = tep_unfair_b + cost_unfair[j-1, i-j]
                cost[j, i-j] = cost[j-1, i-j] + tep_b 
                path[j, i-j,:,:] = np.array([[j-1, i-j], [j, i-j]])

            else:
                cost_unfair[j, i-j] = tep_unfair_a + cost_unfair[j, i-j-1]
                cost[j, i-j] = cost[j, i-j-1] + tep_a 
                path[j, i-j,:,:] = np.array([[j, i-j-1], [j, i-j]])
    return cost, path, cost_unfair


def post_b_score(a_score, b_score, a_label, b_label, lamb = 0, _type="xauc"): ## score has to be decreasing.
    M = len(a_score)
    N = len(b_score)
    if _type == "xauc":
        cost, path_ , cost_unfair = xAUC_post(a_label, b_label, lamb = lamb)
    elif _type=="AUC":
        cost, path_   = maxAUC(a_label, b_label)
    elif _type=="prf":
        cost, path_ , cost_unfair = pairwise_post(a_label, b_label, lamb = lamb)
    else:
        print("Unknown type")
        exit()

    @jit(nopython=True)
    def pathTrace(path):

        trace = []
        tep = path[M,N,:,:]
        trace.append(tep[-1,:])
        trace.append(tep[0,:])
        for i in range(M+N-1):

            tep = path[int(tep[0][0]), int(tep[0][1]), :,:]
            trace.append(tep[0,:])
        trace.reverse()
        return trace

    path = pathTrace(path_)
    gap_a = [[] for i in range(M+1)]

    for i in range(1,len(path)):
        if int(path[i][0])==int(path[i-1][0]):
            gap_a[int(path[i][0])].append(int(path[i][1]))

    changed_b_score  = []
    for bin_ in range(len(gap_a)):
        for item in range(len(gap_a[bin_])):
            num = (len(gap_a[bin_])+1)
            if bin_==0:
                changed_b_score.append((item+1)*a_score[bin_]/num+(num-item-1)/num)
            elif bin_==len(a_score):
                changed_b_score.append((num -item-1)*a_score[bin_-1]/num)
            else:
                changed_b_score.append((item+1)*a_score[bin_]/num + (num-item-1)*a_score[bin_-1]/num)
    if _type=="AUC":
        return np.array(changed_b_score), 0
    else:
        return np.array(changed_b_score), cost_unfair[-1, -1]


def pairwise(a_score, b_score, a_label, b_label):
    sum_ab = 0
    sum_ba = 0
    numa = len(a_label)
    numb = len(b_label)
    a_num1 = np.sum(a_label)
    a_num0 = len(a_label) - a_num1
    b_num1 = np.sum(b_label)
    b_num0 = len(b_label) - b_num1

    i_AUCa = roc_auc_score(a_label, a_score)
    i_AUCb = roc_auc_score(b_label, b_score)

    for i in range(numa):
        for j in range(numb):
            if a_label[i] ==1 and b_label[j] ==0:
                if a_score[i]>b_score[j]:
                    sum_ab+=1
            elif a_label[i]==0 and b_label[j]==1:
                if b_score[j]>a_score[i]:
                    sum_ba+=1
    return (sum_ab+i_AUCa*a_num0*a_num1)/(a_num1*(b_num0+a_num0)), (sum_ba+i_AUCb*b_num0*b_num1)/(b_num1*(a_num0+b_num0))


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


def zeros_mat(a, b):
    a_label = [0] + a
    b_label = [0] + b
    M = len(a_label)-1
    N = len(b_label)-1
    a_1 = np.sum(a)
    b_1 = np.sum(b)
    zeros_mat = np.zeros((M+1, N+1))
    zeros_mat[0,0] = ((N-b_1)+(M - a_1))

    for i in range(1,N+1):
        if b_label[i]==1:
            zeros_mat[0,i] = zeros_mat[0,i-1]
        else:
            zeros_mat[0,i] = zeros_mat[0,i-1]-1 

    for i in range(1,M+1):
        if a_label[i]==0:
            zeros_mat[i,0] = zeros_mat[i-1,0]-1
        else:
            zeros_mat[i,0] = zeros_mat[i-1,0]
        for j in range(1,N+1):
            if b_label[j]==0:
                zeros_mat[i,j] = zeros_mat[i,j-1]-1
            else:
                zeros_mat[i,j] = zeros_mat[i,j-1]
    return zeros_mat






