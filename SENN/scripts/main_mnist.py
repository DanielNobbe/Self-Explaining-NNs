# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
# Ignore Future Warnings (Joosje)
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

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
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader
from tqdm import tqdm


# Local imports
from SENN.arglist import get_senn_parser #parse_args as parse_senn_args
from SENN.models import GSENN
from SENN.conceptizers import image_fcc_conceptizer, image_cnn_conceptizer, input_conceptizer

from SENN.parametrizers import image_parametrizer
from SENN.aggregators import linear_scalar_aggregator, additive_scalar_aggregator
from SENN.trainers import HLearningClassTrainer, VanillaClassTrainer, GradPenaltyTrainer
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid, plot_prob_drop, plot_dependencies
from SENN.eval_utils import estimate_dataset_lipschitz

from robust_interpret.explainers import gsenn_wrapper
from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot, lipschitz_feature_argmax_plot

from random import sample
from tqdm import tqdm
from collections import defaultdict
import itertools


def revert_to_raw(t):
    return ((t*.3081) + .1307)

def load_mnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train = MNIST('data/MNIST', train=True, download=True, transform=transform)
    test  = MNIST('data/MNIST', train=False, download=True, transform=transform)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test

class new_wrapper(gsenn_wrapper):

    def compute_dataset_consistency(self,  dataset, targets = None, reference_value = 0, inputs_are_concepts = True, save_path = None,
     plot_alt_dependencies = True, demo_mode = False):
        """
            does compute_prob_drop for all dataset, returns stats and plots
        """
        drops = []
        atts  = []
        corrs = []
        altcorrs = []
        i = 0
        for x in dataset:
            if demo_mode:
                if i != (demo_mode-1):
                    i += 1
                    continue
            if save_path:
                path = save_path + '_' + str(i) + '/'
            else:
                path = save_path
            target = targets[i]
            p_d, att, deps = self.compute_prob_drop(x, target = target, inputs_are_concepts = inputs_are_concepts,
             save_path = path, alternative = True, plot = True)
            # p_d is now theta*h for each concept.
            p_d = p_d.squeeze()
            att = att.squeeze()
            drops.append(p_d)
            atts.append(att)
            #pdb.set_trace()
            #assert len(p_d).shape[0] == atts.shape[0], "Attributions has wrong size"
            #pdb.set_trace()

            # print("attributions: ", atts)

            corrs.append(np.corrcoef(p_d, att)[0,1])
            # deps, thetas = self.compute_dependencies(x)
            altcorrs.append(np.corrcoef(p_d, deps)[0,1])
            if plot_alt_dependencies:


                classes = ['C' + str(i) for i in (range(1, p_d.shape[0]+1))]
                deps_to_plot = dict(zip(classes, deps))
                thetas_to_plot = dict(zip(classes, att))
                fig, ax = plt.subplots(1, 2)
                A = plot_dependencies(deps_to_plot, title= 'Combined dependencies, target = ' + str(target.item()), sort_rows = False, ax = ax[0])
                B = plot_dependencies(thetas_to_plot, title='Theta dependencies', sort_rows = False, ax = ax[1])
                if not save_path is None:
                    plot_path = save_path + '/dependencies/'
                    if not os.path.isdir(plot_path):
                        os.mkdir(plot_path)
                    fig.savefig(plot_path + str(i) + '.png', format = "png", dpi=300)
                    if demo_mode:
                        plt.close('all')
                        plt.imshow(x[0], cmap='Greys',  interpolation='nearest')
                        plt.savefig(plot_path + str(i) + 'digit.png', format = "png", dpi=300)
            plt.close('all')
            i += 1

        corrs = np.array(corrs)
        altcorrs = np.array(altcorrs)
        # pdb.set_trace()
        # drops = np.stack(drops)
        # atts  = np.stack(atts)
        #
        # np.corrcoef(drops.flatten(), atts.flatten())
        return corrs, altcorrs

    def compute_prob_drop(self, x, target = None, reference_value = 0, plot = False, save_path = None, 
        inputs_are_concepts = True, alternative = False):

        # First, turn inputs into concepts
        if not inputs_are_concepts:
            # x = x.type(torch.FloatTensor)
            x = x.unsqueeze(dim = 0)
            h_x = self.net.forward(x, h_options = -1)
            # print("h_x: ",h_x)
            f = self.net.forward(x, h_x = h_x, h_options = 1)
        # Then, use concepts to forward pass through the model
        # if inputs_are_concepts:
        else:
            f   = self.net.forward(x.reshape(1,-1)) # model is compute_proba function, not neural model - we need to add output layer to compute probabilities. Not necessary for now TODO
        # So for now, we use self.net
        # else:
        #     f = self.model.forward(x.reshape(1,-1), h_x = h_x,  h_options = 1)
        # pred_class = f.argmax()
        if target.nelement()>1:
                _, target = torch.max(target)
        attributions = self(x, y = target) # attributions are theta values (i think)
        deltas = []
        for i in range(h_x.shape[1]):
            x_p = x.clone()
            # x_p[i] = reference_value # uncomment this to be compatible with uci dataset
            h_x_p = h_x.clone()
            h_x_p[:,i, :] = reference_value
            if inputs_are_concepts:
                f_p = self.net(x_p.reshape(1,-1))
            else:
                f_p = self.net.forward(x, h_x = h_x_p,  h_options = 1)
            # print("outcome: ", f_p, f)
            delta_i = (f - f_p)[0,target]
            # print("delta_i: ", delta_i)
            deltas.append(delta_i.cpu().detach().numpy())
        prob_drops = np.array(deltas)

        if not type(attributions) is np.ndarray: #Checks if they are both numpy arrays. If not:
            attributions = attributions.cpu().detach().numpy().squeeze()
        if alternative:
            attributions_plot_alt = attributions.squeeze() * h_x.cpu().detach().numpy().squeeze()
        attributions_plot = attributions.squeeze()
        if plot and not (save_path is None):
            save_path_or = save_path + 'original'
            if not os.path.isdir(save_path):
                # print(save_path)
                os.mkdir(save_path)
            # if not os.path.isdir(save_path_or):
            #     os.mkdir(save_path_or)
            save_path_alt = save_path + 'alternative'
            # if not os.path.isdir(save_path_alt):
            #     os.mkdir(save_path_alt)
            max_att = max(max(attributions_plot) , max(attributions_plot_alt))
            min_att = min(min(attributions_plot) , min(attributions_plot_alt))
            lim_prob_drop = max(max(prob_drops), -min(prob_drops))
            limits = [max_att, min_att, lim_prob_drop]
            plot_prob_drop(attributions_plot.squeeze(), prob_drops, save_path = save_path_or, limits = limits) # remove [0] after attributions for uci

            if alternative:
                plot_prob_drop(attributions_plot_alt.squeeze(), prob_drops, save_path = save_path_alt, limits = limits)

        return prob_drops, attributions, attributions_plot_alt

    # def compute_dependencies(self, x, reference_value = 0, plot = False, save_path = None, inputs_are_concepts = False):
    #     if not inputs_are_concepts:
    #             # x = x.type(torch.FloatTensor)
    #             x = x.unsqueeze(dim = 0)
    #             h_x = self.net.forward(x, h_options = -1)
    #             # print("h_x: ",h_x)
    #             f = self.net.forward(x, h_x = h_x, h_options = 1)
    #         # Then, use concepts to forward pass through the model
    #         # if inputs_are_concepts:
    #     else:
    #         x = h_x # model is compute_proba function, not neural model - we need to add output layer to compute probabilities. Not necessary for now TODO
    #         f   = self.net.forward(x.reshape(1,-1))
    #     thetas = self(x) # attributions are theta values (i think)
    #     dependencies = thetas.squeeze() * h_x.cpu().detach().numpy().squeeze()
    #     return dependencies, thetas


def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpteratbility robustness evaluation on MNIST')

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

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def eval_stability_2(test_tds, expl, scale, our_method=False):
    distances = []  

    for i in tqdm(range(10000)):
        x = Variable(test_tds.dataset[i][0].view(1,1,28,28), volatile = True)
        h_x = expl.net.forward(x, h_options = -1).data.numpy().squeeze()
        
        theta = expl(x)[0]


        if our_method:
            deps = np.multiply(theta, h_x)
        else:
            deps = theta

        # Add noise to sample and repeat
        noise = Variable(scale*torch.randn(x.size()), volatile = True)
        h_x = expl.net.forward(noise, h_options = -1).data.numpy().squeeze()
        theta = expl(noise)[0]
        if our_method:
            deps_noise = np.multiply(theta, h_x)
        else:
            deps_noise = theta

        dist = np.linalg.norm(deps - deps_noise)
        distances.append(dist)

    return distances

def plot_distribution_h(test_tds, expl, plot_type='hx', fig = 0, results_path=False):
    
    values = []
    for i in tqdm(range(10000)):
        x = Variable(test_tds.dataset[i][0].view(1,1,28,28), volatile = True)
        if plot_type == 'hx':
            h_x = expl.net.forward(x, h_options = -1).squeeze().detach().numpy()
            values.append(h_x)
        elif plot_type == 'thetax':
            theta = expl(x)[0]
            values.append(theta)
        elif plot_type == 'thetaxhx':
            h_x = expl.net.forward(x, h_options = -1).squeeze().detach().numpy()
            theta = expl(x)[0]
            values.append(np.multiply(theta, h_x))


    values = list(itertools.chain.from_iterable(values))

    if plot_type == 'hx':
        xtitle = 'Concept values h(x)'
        ytitle = 'p(h(x))'
        plot_color = 'blue'
    elif plot_type == 'thetax':
        xtitle = 'Theta values'
        ytitle = 'p(theta(x))'
        plot_color = 'pink'
    elif plot_type == 'thetaxhx':
        xtitle = 'Theta(x)^T h(x) values'
        ytitle = 'p(theta(x)^T h(x)'
        plot_color = 'purple'

    plt.figure(fig)
    plt.hist(values, color = plot_color, edgecolor = '#CCE6FF', bins=20)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if results_path:
        plt.savefig(results_path + '/histogram' + plot_type + '.png', format = "png", dpi=300)

def main():
    args = parse_args()
    args.nclasses = 10
    args.theta_dim = args.nclasses

   

    model_path, log_path, results_path = generate_dir_names('mnist', args)

    print("Model path out", model_path)

    train_loader, valid_loader, test_loader, train_tds, test_tds = load_mnist_data(
                        batch_size=args.batch_size,num_workers=args.num_workers
                        )

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = 28*28 + int(not args.nobias)
    elif args.h_type == 'cnn':
        #args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_cnn_conceptizer(28*28, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        #args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(28*28, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)


    parametrizer = image_parametrizer(28*28, args.nconcepts, args.theta_dim,  only_positive = args.positive_theta)

    aggregator   = additive_scalar_aggregator(args.concept_dim, args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)

    if args.load_model:
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']

    if args.theta_reg_type in ['unreg','none', None]:
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ = 1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ = 2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ = 3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecoginzed theta_reg_type')

    if not args.load_model and args.train:
        trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
        trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer =  VanillaClassTrainer(model, args)

    trainer.validate(test_loader, fold = 'test')

    All_Results = {}




    ### 1. Single point lipshiz estimate via black box optim
    # All methods tested with BB optim for fair comparison)
    features = None
    classes = [str(i) for i in range(10)]
    model.eval()
    expl = new_wrapper(model,
                        mode      = 'classification',
                        input_type = 'image',
                        multiclass=True,
                        feature_names = features,
                        class_names   = classes,
                        train_data      = train_loader,
                        skip_bias = True,
                        verbose = False)


    ## Faithfulness analysis
    correlations = np.array([])
    altcorrelations = np.array([])
    for i, (inputs, targets) in enumerate(tqdm(test_loader)):
            # get the inputs
            if args.demo:
                if args.nconcepts == 5:
                    if i != 0:
                        continue
                    else:
                        args.demo = 0+1
                elif args.nconcepts == 22:
                    if i != 0:
                        continue
                    else:
                        args.demo = 1+1
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            input_var = torch.autograd.Variable(inputs, volatile=True)
            target_var = torch.autograd.Variable(targets)
            if not args.noplot:
                save_path = results_path + '/faithfulness' + str(i) + '/'
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
            else:
                save_path = None
            corrs, altcorrs = expl.compute_dataset_consistency(input_var, targets = target_var, 
            inputs_are_concepts = False, save_path = save_path, demo_mode = args.demo)
            correlations = np.append(correlations, corrs)
            altcorrelations = np.append(altcorrelations, altcorrs)

    ### Consistency analysis
    correlations = np.array([])
    altcorrelations = np.array([])
    for i, (inputs, targets) in enumerate(tqdm(test_loader)):
            # get the inputs
            if args.demo:
                if args.nconcepts == 5:
                    if i != 0:
                        continue
                    else:
                        args.demo = 27+1
                elif args.nconcepts == 22:
                    if i != 0:
                        continue
                    else:
                        args.demo = 1+1
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            input_var = torch.autograd.Variable(inputs, volatile=True)
            target_var = torch.autograd.Variable(targets)
            if not args.noplot:
                save_path = results_path + '/faithfulness' + str(i) + '/'
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
            else:
                save_path = None
            corrs, altcorrs = expl.compute_dataset_consistency(input_var, targets = target_var, 
            inputs_are_concepts = False, save_path = save_path, demo_mode = args.demo)
            correlations = np.append(correlations, corrs)
            altcorrelations = np.append(altcorrelations, altcorrs)

    average_correlation = np.sum(correlations)/len(correlations)
    std_correlation = np.std(correlations)
    average_alt_correlation = np.sum(altcorrelations)/len(altcorrelations)
    std_alt_correlation = np.std(altcorrelations)
    print("Average correlation:", average_correlation)
    print("Standard deviation of correlations: ", std_correlation)
    print("Average alternative correlation:", average_alt_correlation)
    print("Standard deviation of alternative correlations: ", std_alt_correlation)

    box_plot_values = [correlations, altcorrelations]

    box = plt.boxplot(box_plot_values, patch_artist=True, labels=['theta(x)', 'theta(x) h(x)'])
    colors = ['blue', 'purple']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    print("Figure saved to: ", results_path)
    plt.savefig(results_path + '/faithfulness_box_plot.png', format = "png", dpi=300, verbose=True)


    # Make histograms
    plot_distribution_h(test_loader, expl, 'thetaxhx', fig=0, results_path=results_path)
    plot_distribution_h(test_loader, expl, 'thetax', fig=1, results_path=results_path)
    plot_distribution_h(test_loader, expl, 'hx', fig=2, results_path=results_path)

    # Compute stabilites
    noises = np.arange(0, 0.21, 0.02)
    dist_dict, dist_dict_2 = {}, {}
    for noise in noises:
        distances = eval_stability_2(test_loader, expl, noise, False)
        distances_2 = eval_stability_2(test_loader, expl, noise, True)
        dist_dict[noise] = distances
        dist_dict_2[noise] = distances_2

    # Plot stability
    distances, distances_2, noises = dist_dict, dist_dict_2, noises
    means = [np.mean(distances[noise]) for noise in noises]
    stds = [np.std(distances[noise]) for noise in noises]

    means_min = [means[i] - stds[i] for i in range(len(means))]
    means_max = [means[i] + stds[i] for i in range(len(means))]

    means_2 = [np.mean(distances_2[noise]) for noise in noises]
    stds_2 = [np.std(distances_2[noise]) for noise in noises]

    means_min_2 = [means_2[i] - stds_2[i] for i in range(len(means_2))]
    means_max_2 = [means_2[i] + stds_2[i] for i in range(len(means_2))]

    fig, ax = plt.subplots(1)

    ax.plot(noises, means, lw=2, label='theta(x)', color='blue')
    ax.plot(noises, means_2, lw=2, label='theta(x)^T h(x)', color='purple')
    ax.fill_between(noises, means_max, means_min, facecolor='blue', alpha=0.3)
    ax.fill_between(noises, means_max_2, means_min_2, facecolor='purple', alpha=0.3)
    ax.set_title('Stability')
    ax.legend(loc='upper left')
    ax.set_xlabel('Added noise')
    ax.set_ylabel('Norm of relevance coefficients')
    ax.grid()
    fig.savefig(results_path + '/stablity' + '.png', format = "png", dpi=300)

if __name__ == '__main__':
    main()
