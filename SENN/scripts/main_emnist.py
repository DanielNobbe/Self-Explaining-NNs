# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
import matplotlib.pyplot as plt

# Torch-related
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import torch.utils.data.dataloader as dataloader

# Local imports
from SENN.arglist import get_senn_parser #parse_args as parse_senn_args
from SENN.models import GSENN
from SENN.conceptizers import image_fcc_conceptizer, image_cnn_conceptizer, input_conceptizer

from SENN.parametrizers import image_parametrizer
from SENN.aggregators import linear_scalar_aggregator, additive_scalar_aggregator
from SENN.trainers import HLearningClassTrainer, VanillaClassTrainer, GradPenaltyTrainer
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid
from SENN.eval_utils import estimate_dataset_lipschitz

from robust_interpret.explainers import gsenn_wrapper
from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot

# Download data
from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
EMNIST = list_datasets()

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        # print(self.list_IDs[0])
        # print(self.labels[0])
        # ID = self.list_IDs[0]
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]
        # print('x and y',X, y)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # ID = self.list_IDs[index]
        #
        # # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]
        return self.list_IDs[0], self.labels[0]

def load_emnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    images_train, labels_train = extract_training_samples('letters')
    images_test, labels_test = extract_test_samples('letters')

    train = Dataset(images_train, labels_train)
    test = Dataset(images_test, labels_test)
    num_train = len(train)

    # num_train = images_train.shape[0]
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, valid_loader, test_loader, train, test

def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpretability robustness evaluation on EMNIST')

    # #setup
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                        'glass','diabetes','yeast','leukemia','abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    #####
    args = parser.parse_args()
    # args.nclasses = 26

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args

def main():
    args = parse_args()
    args.nclasses = 26
    args.h_type = 'cnn'
    args.theta_dim = args.nclasses

    model_path, log_path, results_path = generate_dir_names('emnist', args)
    print(args)
    print('model path', model_path)
    print('log path', log_path)
    print('results path', results_path)
    # load_emnist_data(
    #     batch_size=args.batch_size, num_workers=args.num_workers
    # )
    train_loader, valid_loader, test_loader, train_tds, test_tds = load_emnist_data(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # args.h_type == 'cnn':
    conceptizer = image_cnn_conceptizer(28 * 28, args.nconcepts, args.concept_dim)  # , sparsity = sparsity_l)

    parametrizer = image_parametrizer(28*28, args.nconcepts, args.theta_dim,  only_positive = args.positive_theta)

    aggregator   = additive_scalar_aggregator(args.concept_dim, args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)
    if args.load_model:   #false
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']

    # args.theta_reg_type == 'grad3'
    trainer = GradPenaltyTrainer(model, args, typ=3)

    if not args.load_model and args.train:
        # class 'torch.utils.data.dataloader.DataLoader'>

        trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
        # trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'),
                                map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer = VanillaClassTrainer(model, args)

if __name__ == '__main__':
    main()