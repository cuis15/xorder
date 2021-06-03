import numpy as np
import pandas as pd
import random
import sys
from sklearn import model_selection
import tensorflow as tf
from utils import *
import tensorflow_constrained_optimization as tfco
import pdb
import pickle
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
import itertools
import warnings
import argparse
warnings.filterwarnings('ignore')

# Set random seed for reproducibility.
random.seed(333333)
np.random.seed(121212)
tf.random.set_seed(212121)

# We will divide the data into 25 minibatches and refer to them as 'queries'.
# num_queries = 25
def dataset_process(num_queries,run_idx, dataset):
# num_queries = 25*6
    # dataset = 'german' ####size 6000

    fin = open("data/preprocessed/" + dataset + "_data" + '.pkl', 'rb')
    data_dict = pickle.load(fin)
    X, y, a = data_dict["X"], data_dict["y"].astype(np.int), data_dict["a"].astype(np.float32)
    print(y.mean())
    fin = open("data/preprocessed/" + dataset + "_split_idx_" + str(run_idx) + '.pkl', 'rb')
    data_dict = pickle.load(fin)
    idx_train, idx_test = data_dict["idx_train"], data_dict["idx_test"]

    queries = np.random.randint(0, num_queries, size=X.shape[0])

    train_set = {
      'features': X[idx_train, :],
      'labels': y[idx_train],
      'groups': a[idx_train],
      'dimension': X.shape[-1],
      'queries': queries[idx_train],
      'num_queries': num_queries
    }

    # Test features, labels and protected groups.
    test_set = {
      'features': X[idx_test, :],
      'labels': y[idx_test],
      'groups': a[idx_test],
      'dimension': X.shape[-1],
      'queries': queries[idx_train],
      'num_queries': num_queries
    }

    return train_set, test_set
#######################################################################################

def pair_high_low_docs(data):
  # Returns a DataFrame of pairs of larger-smaller labeled regression examples
  # given in DataFrame.
  # For all pairs of docs, and remove rows that are not needed.
  pos_docs = data.copy()
  neg_docs = data.copy()

  # Include a merge key.
  pos_docs.insert(0, "merge_key", 0)
  neg_docs.insert(0, "merge_key", 0)

  # Merge docs and drop merge key and label column.
  pairs = pos_docs.merge(neg_docs, on="merge_key", how="outer",
                         suffixes=("_pos", "_neg"))

  # Only retain rows where label_pos > label_neg.
  pairs = pairs[pairs.label_pos > pairs.label_neg]

  # Drop merge_key.
  pairs.drop(columns=["merge_key"], inplace=True)
  return pairs


def convert_labeled_to_paired_data(data_dict, index=None):
  # Forms pairs of examples from each batch/query.

  # Converts data arrays to pandas DataFrame with required column names and
  # makes a call to convert_df_to_pairs and returns a dictionary.
  features = data_dict['features']
  labels = data_dict['labels']
  groups = data_dict['groups']
  queries = data_dict['queries']

  if index is not None:
    data_df = pd.DataFrame(features[queries == index, :])
    data_df = data_df.assign(label=pd.DataFrame(labels[queries == index]))
    data_df = data_df.assign(group=pd.DataFrame(groups[queries == index]))
    data_df = data_df.assign(query_id=pd.DataFrame(queries[queries == index]))
  else:
    data_df = pd.DataFrame(features)
    data_df = data_df.assign(label=pd.DataFrame(labels))
    data_df = data_df.assign(group=pd.DataFrame(groups))
    data_df = data_df.assign(query_id=pd.DataFrame(queries))

  # Forms pairs of positive-negative docs for each query in given DataFrame
  # if the DataFrame has a query_id column. Otherise forms pairs from all rows
  # of the DataFrame.
  data_pairs = data_df.groupby('query_id').apply(pair_high_low_docs)

  # Create groups ndarray.
  pos_groups = data_pairs['group_pos'].values.reshape(-1, 1)
  neg_groups = data_pairs['group_neg'].values.reshape(-1, 1)
  group_pairs = np.concatenate((pos_groups, neg_groups), axis=1)

  # Create queries ndarray.
  query_pairs = data_pairs['query_id_pos'].values.reshape(-1,)

  # Create features ndarray.
  feature_names = data_df.columns
  feature_names = feature_names.drop(['query_id', 'label'])
  feature_names = feature_names.drop(['group'])

  pos_features = data_pairs[[str(s) + '_pos' for s in feature_names]].values
  pos_features = pos_features.reshape(-1, 1, len(feature_names))

  neg_features = data_pairs[[str(s) + '_neg' for s in feature_names]].values
  neg_features = neg_features.reshape(-1, 1, len(feature_names))

  feature_pairs = np.concatenate((pos_features, neg_features), axis=1)

  # Paired data dict.
  paired_data = {
      'feature_pairs': feature_pairs,
      'group_pairs': group_pairs,
      'query_pairs': query_pairs,
      'features': features,
      'labels': labels,
      'queries': queries,
      'dimension': data_dict['dimension'],
      'num_queries': data_dict['num_queries']
  }

  return paired_data


def get_mask(groups, pos_group, neg_group=None):
    # Returns a boolean mask selecting positive-negative document pairs where
    # the protected group for  the positive document is pos_group and
    # the protected group for the negative document (if specified) is neg_group.
    # Repeat group membership positive docs as many times as negative docs.
    mask_pos = groups[:, 0] == pos_group

    if neg_group is None:
        return mask_pos
    else:
        mask_neg = groups[:, 1] == neg_group
        return mask_pos & mask_neg


def mean_squared_error(model, dataset):
    # Returns mean squared error for Keras model on dataset.
    scores = model.predict(dataset['features'])
    labels = dataset['labels']
    cross_entropy = labels * np.log(scores) + (1 - labels) * np.log(1 - scores)
    return np.mean(cross_entropy)


def group_error_rate(model, dataset, pos_group, neg_group=None):
    # Returns error rate for Keras model on data set, considering only document
    # pairs where the protected group for the positive document is pos_group, and
    # the protected group for the negative document (if specified) is neg_group.
    d = dataset['dimension']
    scores0 = model.predict(dataset['feature_pairs'][:, 0, :].reshape(-1, d))
    scores1 = model.predict(dataset['feature_pairs'][:, 1, :].reshape(-1, d))
    mask = get_mask(dataset['group_pairs'], pos_group, neg_group)
    diff = scores0 - scores1
    diff = diff[mask > 0].reshape((-1))
    return np.mean(diff < 0)


def create_scoring_model(feature_pairs, features, dimension):
    # Returns a linear Keras scoring model, and returns a nullary function
    # returning predictions on the features.

    # Linear scoring model with no hidden layers.
    layers = []
    # Input layer takes `dimension` inputs.
    layers.append(tf.keras.Input(shape=(dimension,)))
    layers.append(tf.keras.layers.Dense(1, kernel_initializer='zeros',bias_initializer='zeros',activation = 'sigmoid'))
    # layers.append(tf.keras.layers.Dense(1))
    scoring_model = tf.keras.Sequential(layers)

    # Create a nullary function that returns applies the linear model to the
    # features and returns the tensor with the prediction differences on pairs.
    def prediction_diffs():
        scores0 = scoring_model(feature_pairs()[:, 0, :].reshape(-1, dimension))
        scores1 = scoring_model(feature_pairs()[:, 1, :].reshape(-1, dimension))
        return scores0 - scores1

    # Create a nullary function that returns the predictions on individual
    # examples.
    predictions = lambda: scoring_model(features())

    return scoring_model, prediction_diffs, predictions


def group_mask_fn(groups, pos_group, neg_group=None):
    # Returns a nullary function returning group mask.
    group_mask = lambda: np.reshape(
        get_mask(groups(), pos_group, neg_group), (-1))
    return group_mask


def formulate_problem(
        feature_pairs, group_pairs, features, labels, dimension,
        constraint_groups=[], constraint_slack=None):
    # Formulates a constrained problem that optimizes the squared error for a linear
    # model on the specified dataset, subject to pairwise fairness constraints
    # specified by the constraint_groups and the constraint_slack.
    #
    # Args:
    #   feature_pairs: Nullary function returning paired features
    #   group_pairs: Nullary function returning paired groups
    #   features: Nullary function returning features
    #   labels: Nullary function returning labels
    #   dimension: Input dimension for scoring model
    #   constraint_groups: List containing tuples of the form
    #     ((pos_group0, neg_group0), (pos_group1, neg_group1)), specifying the
    #     group memberships for the document pairs to compare in the constraints.
    #   constraint_slack: slackness '\epsilon' allowed in the constraints.
    # Returns:
    #   A RateMinimizationProblem object, and a Keras scoring model.

    # Create linear scoring model: we get back a Keras model and a nullary
    # function returning predictions on the features.
    scoring_model, prediction_diffs, predictions = create_scoring_model(
        feature_pairs, features, dimension)

    # Context for the optimization objective.
    context = tfco.rate_context(prediction_diffs)

    # Squared loss objective.
    # squared_loss = lambda: tf.reduce_mean((predictions() - labels()) ** 2)
    squared_loss = lambda: tf.reduce_mean(-(labels() * tf.math.log(predictions()) + (1-labels()) * tf.math.log(1-predictions())))

    # Constraint set.
    constraint_set = []

    # Context for the constraints.
    for ((pos_group0, neg_group0), (pos_group1, neg_group1)) in constraint_groups:
        # Context for group 0.
        group_mask0 = group_mask_fn(group_pairs, pos_group0, neg_group0)
        context_group0 = context.subset(group_mask0)

        # Context for group 1.
        group_mask1 = group_mask_fn(group_pairs, pos_group1, neg_group1)
        context_group1 = context.subset(group_mask1)

        # Add constraints to constraint set.
        constraint_set.append(
            tfco.negative_prediction_rate(context_group0) <= (
                    tfco.negative_prediction_rate(context_group1) + constraint_slack))
        constraint_set.append(
            tfco.negative_prediction_rate(context_group1) <= (
                    tfco.negative_prediction_rate(context_group0) + constraint_slack))

    # Formulate constrained minimization problem.
    problem = tfco.RateMinimizationProblem(
        tfco.wrap_rate(squared_loss), constraint_set)

    return problem, scoring_model


def train_model(train_set, params):
    # Trains the model with stochastic updates (one query per updates).
    #
    # Args:
    #   train_set: Dictionary of "paired" training data.
    #   params: Dictionary of hyper-paramters for training.
    #
    # Returns:
    #   Trained model, list of objectives, list of group constraint violations.

    # Set up problem and model.
    if params['constrained']:
        # Constrained optimization.
        if params['constraint_type'] == 'marginal_equal_opportunity':
            constraint_groups = [((0, None), (1, None))]
        elif params['constraint_type'] == 'cross_group_equal_opportunity':
            constraint_groups = [((0, 1), (1, 0))]
        else:
            constraint_groups = [((0, 1), (1, 0)), ((0, 0), (1, 1))]
    else:
        # Unconstrained optimization.
        constraint_groups = []

    # Dictionary that will hold batch features pairs, group pairs and labels for
    # current batch. We include one query per-batch.
    paired_batch = {}
    batch_index = 0  # Index of current query.

    # Data functions.
    feature_pairs = lambda: paired_batch['feature_pairs']
    group_pairs = lambda: paired_batch['group_pairs']
    features = lambda: paired_batch['features']
    labels = lambda: paired_batch['labels']

    # Create scoring model and constrained optimization problem.
    problem, scoring_model = formulate_problem(
        feature_pairs, group_pairs, features, labels, train_set['dimension'],
        constraint_groups, params['constraint_slack'])

    # Create a loss function for the problem.
    lagrangian_loss, update_ops, multipliers_variables = (
        tfco.create_lagrangian_loss(problem, dual_scale=params['dual_scale']))

    # Create optimizer
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=params['learning_rate'])

    # List of trainable variables.
    var_list = (
            scoring_model.trainable_weights + problem.trainable_variables +
            [multipliers_variables])

    # List of objectives, group constraint violations.
    # violations, and snapshot of models during course of training.
    objectives = []
    group_violations = []
    models = []

    feature_pair_batches = train_set['feature_pairs']
    group_pair_batches = train_set['group_pairs']
    query_pairs = train_set['query_pairs']
    feature_batches = train_set['features']
    label_batches = train_set['labels']
    queries = train_set['queries']

    print()
    # Run loops * iterations_per_loop full batch iterations.
    for ii in range(params['loops']):
        for jj in range(params['iterations_per_loop']):
            # Populate paired_batch dict with all pairs for current query. The batch
            # index is the same as the current query index.
            paired_batch = {
                'feature_pairs': feature_pair_batches[query_pairs == batch_index],
                'group_pairs': group_pair_batches[query_pairs == batch_index],
                'features': feature_batches[queries == batch_index],
                'labels': label_batches[queries == batch_index]
            }

            # Optimize loss.
            update_ops()
            optimizer.minimize(lagrangian_loss, var_list=var_list)

            # Update batch_index, and cycle back once last query is reached.
            batch_index = (batch_index + 1) % train_set['num_queries']

        # Snap shot current model.
        model_copy = tf.keras.models.clone_model(scoring_model)
        model_copy.set_weights(scoring_model.get_weights())
        models.append(model_copy)

        # Evaluate metrics for snapshotted model.
        error, gerr, group_viol = evaluate_results(
            scoring_model, train_set, params)
        objectives.append(error)
        group_violations.append(
            [x - params['constraint_slack'] for x in group_viol])

        sys.stdout.write(
            '\r Loop %d: error = %.3f, max constraint violation = %.3f' %
            (ii, objectives[-1], max(group_violations[-1])))
    print()

    if params['constrained']:
        # Find model iterate that trades-off between objective and group violations.
        best_index = tfco.find_best_candidate_index(
            np.array(objectives), np.array(group_violations), rank_objectives=False)
    else:
        # Find model iterate that achieves lowest objective.
        best_index = np.argmin(objectives)

    return models[best_index]


def evaluate_results(model, test_set, params):
    # Returns sqaured error, group error rates, group-level constraint violations.
    if params['constraint_type'] == 'marginal_equal_opportunity':
        g0_error = group_error_rate(model, test_set, 0)
        g1_error = group_error_rate(model, test_set, 1)
        group_violations = [g0_error - g1_error, g1_error - g0_error]
        return (mean_squared_error(model, test_set), [g0_error, g1_error],
                group_violations)
    else:
        g00_error = group_error_rate(model, test_set, 0, 0)
        g01_error = group_error_rate(model, test_set, 0, 1)
        g10_error = group_error_rate(model, test_set, 1, 1)
        g11_error = group_error_rate(model, test_set, 1, 1)
        group_violations_offdiag = [g01_error - g10_error, g10_error - g01_error]
        group_violations_diag = [g00_error - g11_error, g11_error - g00_error]

        if params['constraint_type'] == 'cross_group_equal_opportunity':
            return (mean_squared_error(model, test_set),
                    [[g00_error, g01_error], [g10_error, g11_error]],
                    group_violations_offdiag)
        else:
            return (mean_squared_error(model, test_set),
                    [[g00_error, g01_error], [g10_error, g11_error]],
                    group_violations_offdiag + group_violations_diag)



def show_results(model,f, method = 'Unconstrained'):
    pred_train = model.predict(train_set['features'])[:,0]
    AUC_train = roc_auc_score(train_set['labels'], pred_train)
    pred_test = model.predict(test_set['features'])[:,0]
    AUC_test = roc_auc_score(test_set['labels'], pred_test)
    disparity_train,_ ,_ = cal_fairness_metric(pred_train, train_set['labels'], train_set['groups'], metric='xauc')
    disparity_test,_ ,_ = cal_fairness_metric(pred_test, test_set['labels'], test_set['groups'], metric='xauc')

    print('\nMethod\t\t\tType\t\tAUC\t\tDiff')
    print('%s\t%s\t\t%.3f\t\t%.3f' % (method, 'train', AUC_train, disparity_train))
    print('%s\t%s\t\t%.3f\t\t%.3f' % (method, 'test', AUC_test, disparity_test))
    f.write('\nMethod\t\t\tType\t\tAUC\t\tDiff'+'\t')
    f.write('\n' +'%s\t%s\t\t%.3f\t\t%.3f' % (method, 'train', AUC_train, disparity_train)+'\t')
    f.write('\n' +'%s\t%s\t\t%.3f\t\t%.3f' % (method, 'test', AUC_test, disparity_test)+'\t')



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default="german", help="dataset to run(compas, framingham, adult, german)")
parser.add_argument('--eval_metric', type = str, default="prf", help="metric of ranking fairness, xauc or prf")
args = parser.parse_args()


dataset = args.dataset
eval_metric = args.eval_metric


if eval_metric == 'xauc':
    lambs_xorder = [0.01 * i for i in range(15)]
else:
    lambs_xorder = [0.01 * i for i in range(15)]

epsilons = [0.0,0.05,0.1,0.15,0.2,0.25]
num_run = 2

aucs_con = np.zeros((len(epsilons), num_run))
disparity_train_con = np.zeros((len(epsilons), num_run))
disparity_test_con = np.zeros((len(epsilons), num_run))

aucs_un = np.zeros(num_run)
disparity_train_un = np.zeros(num_run)
disparity_test_un = np.zeros(num_run)

aucs_log = np.zeros(num_run)
disparity_train_log = np.zeros(num_run)
disparity_test_log = np.zeros(num_run)

aucs_xorder = np.zeros((len(lambs_xorder), num_run))
disparity_train_xorder = np.zeros((len(lambs_xorder), num_run))
disparity_test_xorder = np.zeros((len(lambs_xorder), num_run))

for run_idx in range(num_run):
    for num_queries in [4*1]:
        train_set, test_set = dataset_process(num_queries,run_idx, dataset)

    # Convert train/test set to paired data for later evaluation.
        paired_train_set = convert_labeled_to_paired_data(train_set)
        paired_test_set = convert_labeled_to_paired_data(test_set)

    # Model hyper-parameters.


        with open("results_log", 'a') as f:
            param_pool = OrderedDict([('loops', [20]),
        ('iterations_per_loop', [100]),
        ('learning_rate', [0.001]),
        ('constraint_type', ['marginal_equal_opportunity']),
        ('constraint_slack', [0.02]),
        ('dual_scale', [1 ])
        ])

            for values in itertools.product(*param_pool.values()):
                keys = param_pool.keys()
                task_param = dict(list(zip(keys, values)))
                print(task_param)
                f.write(str(task_param)+ '\n')

                if eval_metric == 'xauc':
                    task_param['constraint_type'] = 'cross_group_equal_opportunity'
                else:
                    task_param['constraint_type'] = 'marginal_equal_opportunity'

            # Unconstrained optimization.
                task_param['constrained'] = False
                model_unc  = train_model(paired_train_set, task_param)
                show_results(model_unc,f, method = 'Unconstrained')

                pred_train_un = model_unc.predict(train_set['features'])[:, 0]
                pred_test_un = model_unc.predict(test_set['features'])[:, 0]

                y_train, a_train = train_set['labels'], train_set['groups']
                y_test, a_test = test_set['labels'], test_set['groups']

                auc_test = roc_auc_score(y_test, pred_test_un)
                auc_train = roc_auc_score(y_train, pred_train_un)
                disparity_train, _, _ = cal_fairness_metric(pred_train_un, y_train, a_train, metric=eval_metric)
                disparity_test, _, _ = cal_fairness_metric(pred_test_un, y_test, a_test, metric=eval_metric)
                print(auc_train, auc_test, disparity_train, disparity_test)
                aucs_un[run_idx] = auc_test
                disparity_train_un[run_idx] = disparity_train
                disparity_test_un[run_idx] = disparity_test

                # # Sorting the instances of group a and b
                tr_a_score_sort, tr_b_score_sort, tr_a_label_sort, tr_b_label_sort = generate_sorted_groups(
                    pred_train_un, y_train,
                    a_train)
                print()
                te_a_score_sort, te_b_score_sort, te_a_label_sort, te_b_label_sort = generate_sorted_groups(
                    pred_test_un, y_test,
                    a_test)

                print("Running post-log...")
                beta = -2.0
                paras, disparities_train = [], []
                # Searching on the space of \alpha with fixed \beta, this is the same as in the supplemental material of post-logit
                for a_idx in range(100):
                    alpha = 0.1 * a_idx
                    adjust_tr_b_score_sort = 1 / (1 + np.exp(-(alpha * tr_b_score_sort + beta)))
                    disparity_train, _, _ = cal_fairness_metric_by_groups(tr_a_score_sort, adjust_tr_b_score_sort,
                                                                          tr_a_label_sort,
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
                print(auc_test, disparity_train, disparity_test)
                aucs_log[run_idx] = auc_test
                disparity_train_log[run_idx] = disparity_train
                disparity_test_log[run_idx] = disparity_test

                print("Running xorder...")
                k = y_train.sum() * (1 - y_train).sum()
                for (lamb_idx, lamb) in enumerate(lambs_xorder):
                    post_tr_b_score, _ = post_b_score(tr_a_score_sort, tr_b_score_sort,
                                                      np.concatenate(([0], tr_a_label_sort), axis=0),
                                                      np.concatenate(([0], tr_b_label_sort), axis=0), lamb * k,
                                                      _type=eval_metric)
                    post_te_b_score = post_score(tr_b_score_sort, post_tr_b_score, te_b_score_sort)

                    post_auc = roc_auc_score(list(te_a_label_sort) + list(te_b_label_sort),
                                             list(te_a_score_sort) + list(post_te_b_score))

                    _, m_ab_tr, m_ba_tr = cal_fairness_metric_by_groups(tr_a_score_sort, post_tr_b_score,
                                                                        tr_a_label_sort,
                                                                        tr_b_label_sort, eval_metric)
                    _, m_ab_te, m_ba_te = cal_fairness_metric_by_groups(te_a_score_sort, post_te_b_score,
                                                                        te_a_label_sort,
                                                                        te_b_label_sort, eval_metric)
                    print(lamb, post_auc, abs(m_ab_tr - m_ba_tr), abs(m_ab_te - m_ba_te))
                    disparity_train_xorder[lamb_idx, run_idx] = abs(m_ab_tr - m_ba_tr)
                    disparity_test_xorder[lamb_idx, run_idx] = abs(m_ab_te - m_ba_te)
                    aucs_xorder[lamb_idx, run_idx] = post_auc

                # Constrained optimization with TFCO.
                task_param['constrained'] = True
                for (eps_idx, constraint_slack) in enumerate(epsilons):
                    task_param['constraint_slack'] = constraint_slack
                    model_con = train_model(paired_train_set, task_param)

                    pred_train_con = model_con.predict(train_set['features'])[:, 0]
                    pred_test_con = model_con.predict(test_set['features'])[:, 0]
                    auc_test = roc_auc_score(y_test, pred_test_con)
                    auc_train = roc_auc_score(y_train, pred_train_con)
                    disparity_train, _, _ = cal_fairness_metric(pred_train_con, y_train, a_train, metric=eval_metric)
                    disparity_test, _, _ = cal_fairness_metric(pred_test_con, y_test, a_test, metric=eval_metric)
                    print(auc_train, auc_test, disparity_train, disparity_test)
                    aucs_con[eps_idx, run_idx] = auc_test
                    disparity_train_con[eps_idx, run_idx] = disparity_train
                    disparity_test_con[eps_idx, run_idx] = disparity_test



print("Result of unadjusted:")
print(dataset, eval_metric)
print("Train disparity:{:.3f}".format(disparity_train_un.mean()))
print("Test disparity: {:.3f}".format(disparity_test_un.mean()))
print("Test total AUC: {:.3f}".format(aucs_un.mean()))

print("Result for post-log:")
print("Train disparity:{:.3f}".format(disparity_train_log.mean()))
print("Test disparity: {:.3f}".format(disparity_test_log.mean()))
print("Test total AUC: {:.3f}".format(aucs_log.mean()))

print("Result for xorder under different weights:")
print("Train disparity:", array2str(disparity_train_xorder.mean(1)))
print("Test disparity: ", array2str(disparity_test_xorder.mean(1)))
print("Test total AUC: ", array2str(aucs_xorder.mean(1)))

print("Result for constrains:")
print("Train disparity:", array2str(disparity_train_con.mean(1)))
print("Test disparity: ", array2str(disparity_test_con.mean(1)))
print("Test total AUC: ", array2str(aucs_con.mean(1)))


exit()
