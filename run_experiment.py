import torch
import random
import argparse
from experiment import run_experiment

manualSeed = 0
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default="compas", help="dataset to run(compas, framingham, adult, german)")
parser.add_argument('--eval_metric', type = str, default="xauc", help="metric of ranking fairness, xauc or prf")
parser.add_argument('--classifier', type = str, default="lr", help="classificaion model. lr for logistic regression, rb for rankboost")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset, eval_metric, classifier = args.dataset, args.eval_metric, args.classifier
    print("Run experiment for classifier {}, metric {} on {} dataset".format(classifier,eval_metric,dataset))
    run_experiment(dataset, eval_metric, classifier)

