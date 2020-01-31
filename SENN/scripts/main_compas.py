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

import sys, os
import numpy as np
import pdb
import pandas as pd
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from os.path import dirname, realpath
from SENN.arglist import parse_args
from SENN.utils import plot_theta_stability, generate_dir_names, plot_dependencies, plot_prob_drop

from SENN.models import GSENN
from SENN.conceptizers import input_conceptizer, image_fcc_conceptizer
from SENN.parametrizers import  dfc_parametrizer
from SENN.aggregators import additive_scalar_aggregator
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer
import itertools

from robust_interpret.explainers import gsenn_wrapper

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')

import random

class new_wrapper(gsenn_wrapper):

    def plot_distribution_h(self, test_tds, plot_type='h(x)'):
    
        values = []
        for i in tqdm(range(len(test_tds.dataset))):
            x = Variable(test_tds.dataset[i][0], volatile = True)
            if plot_type == 'hx':
                h_x = x.data.numpy().squeeze()
                values.append(h_x)
            elif plot_type == 'thetax':
                f   = self.net.forward(x.reshape(1,-1))
                pred_class = f.argmax()
                theta = self(x, y = pred_class).squeeze()
                values.append(theta)
            elif plot_type == 'thetaxhx':
                h_x = x.data.numpy().squeeze()
                f   = self.net.forward(x.reshape(1,-1))
                pred_class = f.argmax()
                theta = self(x, y = pred_class).squeeze()
                values.append(np.multiply(theta, h_x))


        values = list(itertools.chain.from_iterable(values))

        return values

    def compute_stability(self, dataset, sigma = 0.1):
        distances_1 = []
        distances_2 = []

        for i in range(len(dataset.dataset)):

            x = Variable(dataset.dataset[i][0], volatile = True)
            h_x = x.data.numpy().squeeze()
            f   = self.net.forward(x.reshape(1,-1))
            pred_class = f.argmax()
            theta = self(x, y = pred_class).squeeze()

            deps_1 = theta
            deps_2 = np.multiply(theta, h_x)

            # ADD NOISE
            flip = random.choice(range(6))

            # Flip two_yr_recisivism
            if flip == 0:
                if x[0] == float(0):
                    x[0] = float(1)
                else:
                    x[0] == float(0)
            # Add noise to number of priors
            elif flip == 1:
                x[1] = x[1] + np.random.normal(0, sigma)
            # Flip age
            elif flip == 2:
                if x[2] == float(1.0):
                    x[2], x[3] == float(0), float(1)
                else:
                    x[3], x[2] == float(0), float(1)
            # Flip race
            elif flip == 3:
                select = True
                race = (x[4:9] == float(1.0)).nonzero()
                if len(race) == 0:
                    race = 100
                else:
                    race = race[0] + 4
                while select:
                    index = random.choice(range(4, 9))
                    if index != race:
                        select = False
                x[4:9] = float(0)
                x[index] = float(1)
            # Flip gender
            elif flip == 4:
                if x[9] == float(0.0):
                    x[9] = float(1.0)
                else:
                    x[9] = float(0.0)
            elif flip == 5:
                if x[10] == float(0.0):
                    x[10] = float(1.0)
                else:
                    x[10] = float(0.0)

            h_x = x.data.numpy().squeeze()
            f = self.net.forward(x.reshape(1,-1))
            pred_class = f.argmax()
            theta = self(x, y = pred_class).squeeze()

            deps_1_noise = theta
            deps_2_noise = np.multiply(theta, h_x)

            # Compute norms
            dist_1 = np.linalg.norm(deps_1 - deps_1_noise)
            distances_1.append(dist_1)

            dist_2 = np.linalg.norm(deps_2 - deps_2_noise)
            distances_2.append(dist_2)

        return distances_1, distances_2


    def compute_dataset_consistency(self,  dataset, targets = None,
     reference_value = 0, inputs_are_concepts = True, save_path = None,
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
                if i != demo_mode-1:
                    i += 1
                    continue
            if save_path:
                path = save_path + '_' + str(i) + '/'
            else:
                path = save_path
            if targets is not None:
                target = targets[i]
            else:
                target = None
            p_d, att, deps = self.compute_prob_drop(x, target = target, save_path = path,
              alternative = True, inputs_are_concepts = True)
            # p_d is now theta*h for each concept.
            p_d = p_d.squeeze()
            att = att.squeeze()
            drops.append(p_d)
            atts.append(att)
            #pdb.set_trace()
            #assert len(p_d).shape[0] == atts.shape[0], "Attributions has wrong size"
            #pdb.set_trace()

            
            if not np.all(p_d == 0):
                corrs.append(np.corrcoef(p_d, att)[0,1])
                # deps, thetas = self.compute_dependencies(x, inputs_are_concepts = True)
                altcorrs.append(np.corrcoef(p_d, deps)[0,1])
                if plot_alt_dependencies and (not path is None):


                    classes = ['C' + str(i) for i in range(p_d.shape[0])]
                    deps_to_plot = dict(zip(classes, deps))
                    thetas_to_plot = dict(zip(classes, atts[0]))
                    fig, ax = plt.subplots(1, 2)
                    if targets is not None:
                        title = 'Combined dependencies, target = ' + str(target.item())
                    else:
                        title = 'Combined dependencies'
                    A = plot_dependencies(deps_to_plot, title= title , sort_rows = False, ax = ax[0])
                    B = plot_dependencies(thetas_to_plot, title='Theta dependencies', sort_rows = False, ax = ax[1])
                    
                    if demo_mode:
                        print("\n Input values: ", x)

                    plot_path = path + '/dependencies/'
                    if not os.path.isdir(plot_path):
                        os.mkdir(plot_path)
                    fig.savefig(plot_path + str(i) + '.png', format = "png", dpi=300)
            plt.close('all')
            i += 1
        corrs = np.array(corrs)
        altcorrs = np.array(altcorrs)

        return corrs, altcorrs

    def compute_prob_drop(self, x, target = None, reference_value = 0, plot = False, 
    save_path = None, alternative = False, inputs_are_concepts = False):
        """
            This is placed here to prevent having to update the robust_interpret package.
        """

        f   = self.net.forward(x.reshape(1,-1)) 
        h_x = x

        pred_class = f.argmax()

        attributions = self(x, y = target) # attributions are theta values (i think)
        deltas = []
        for i in range(h_x.shape[0]):
            x_p = x.clone()
            x_p[i] = reference_value 
            h_x_p = h_x.clone()
            h_x_p[i] = reference_value
            if inputs_are_concepts:
                f_p = self.net(x_p.reshape(1,-1))
            else:
                f_p = self.net.forward(x, h_x = h_x_p,  h_options = 1)
            delta_i = (f - f_p)[0,pred_class]
            deltas.append(delta_i.cpu().detach().numpy())
        prob_drops = np.array(deltas)
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().detach().numpy()
        if alternative:
            attributions_plot_alt = attributions.squeeze() * h_x.cpu().detach().numpy().squeeze()
        attributions_plot = attributions.squeeze()
        plot = True
        if plot and not save_path is None:
            save_path_or = save_path + 'original'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            save_path_alt = save_path + 'alternative'

            ex_att = max( max(max(attributions_plot) , max(attributions_plot_alt)), 
            -min(min(attributions_plot) , min(attributions_plot_alt)))
            lim_prob_drop = max(max(prob_drops), -min(prob_drops))
            limits = [ex_att, lim_prob_drop]
            plot_prob_drop(attributions_plot.squeeze(), prob_drops, save_path = save_path_or, limits=limits) # remove [0] after attributions for uci
            if alternative:
                plot_prob_drop(attributions_plot_alt.squeeze(), prob_drops, save_path = save_path_alt, limits=limits)
        return prob_drops, attributions, attributions_plot_alt



def find_conflicting(df, labels, consensus_delta = 0.2):
    """
        Find examples with same exact feat vector but different label.
        Finds pairs of examples in dataframe that differ only
        in a few feature values.

        Args:
            - differ_in: list of col names over which rows can differ
    """
    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in tqdm(range(len(df))):
        if full_dups[i] and (not i in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df  = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5)< consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)

def load_compas_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size=64):
    df= pd.read_csv("data/propublica_data_for_fairml.csv")
    df['Number_of_Priors'] = np.sqrt(df['Number_of_Priors'])/(np.sqrt(38))
    compas_rating = df.score_factor.values 
    df = df.drop("score_factor", 1)

    pruned_df, pruned_rating = find_conflicting(df, compas_rating)
    x_train, x_test, y_train, y_test   = train_test_split(pruned_df, pruned_rating, test_size=0.1, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)

    feat_names = list(x_train.columns)
    print("feature names", feat_names)
    x_train = x_train.values 
    x_test  = x_test.values

    Tds = []
    Loaders = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler(with_std=False, with_mean=False) 
        transformed = scaler.fit_transform(foldx)
        tds = TensorDataset(torch.from_numpy(transformed).float(),
                            torch.from_numpy(foldy).view(-1, 1).float())
        loader = DataLoader(tds, batch_size=batch_size, shuffle=False)
        Tds.append(tds)
        Loaders.append(loader)

    return (*Loaders, *Tds, df, feat_names)

def main():
    args = parse_args()
    args.nclasses = 1
    args.theta_dim = args.nclasses
    args.print_freq = 100
    args.epochs = 10
    train_loader, valid_loader, test_loader, train, valid, test, data, feat_names  = load_compas_data()

    layer_sizes = (10,10,5)
    input_dim = 11

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = 11 + int(not args.nobias)
    elif args.h_type == 'fcc':
        args.nconcepts += int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(11, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        raise ValueError('Unrecognized h_type')

    model_path, log_path, results_path = generate_dir_names('compas', args)


    parametrizer = dfc_parametrizer(input_dim, *layer_sizes, args.nconcepts, args.theta_dim)

    aggregator   = additive_scalar_aggregator(args.concept_dim,args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator)

    if args.theta_reg_type == 'unreg':
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
        raise ValueError('Unrecognized theta_reg_type')

    if not args.load_model and args.train:
        trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
        trainer.plot_losses(save_path=results_path)

    # Load Best One
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'))
    model = checkpoint['model']

    results = {}

    # set up input arguments for GSENN wrapper
    features = None
    classes = [0, 1]

    #
    expl = new_wrapper(model,
                        mode = 'classification',
                        input_type = 'feature',
                        feature_names = 'features',
                        class_names = 'classes',
                        train_data = train_loader,
                        skip_bias = True,
                        verbose = False)

    ### Faithfulness analysis

    print("Performing faithfulness analysis...")


    correlations = np.array([])
    altcorrelations = np.array([])
    for i, (inputs, targets) in enumerate(tqdm(test_loader)):

            if args.demo:
                if i != 0:
                    continue
                else:
                    args.demo = 0 + 1
            input_var = torch.autograd.Variable(inputs, volatile=True)
            target_var = torch.autograd.Variable(targets)
            if not args.noplot:
                save_path = results_path + '/faithfulness' + str(i) + '/'
                if not os.path.isdir(save_path):
                    os.mkdir(save_path) 
            else:
                save_path = None
            corrs, altcorrs = expl.compute_dataset_consistency(input_var, targets = target_var, 
            inputs_are_concepts = False, save_path = save_path, demo_mode=args.demo)
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

    print("Generating Faithfulness correlation box plot..")

    box_plot_values = [correlations, altcorrelations]
    plt.close('all')
    box = plt.boxplot(box_plot_values, patch_artist=True, labels=['theta(x)', 'theta(x) h(x)'])
    colors = ['blue', 'purple']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.savefig(results_path + '/faithfulness_box_plot.png', format = "png", dpi=300)

    if args.noplot:
        stabilities_1, stabilities_2 = expl.compute_stability(test_loader)
        stabilities = [stabilities_1, stabilities_2]
        plt.close('all')
        box = plt.boxplot(stabilities, patch_artist=True, labels=['theta(x)', 'theta(x) h(x)'])
        colors = ['blue', 'purple']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.savefig(results_path + '/stability' + '.png', format = "png", dpi=300)

    if args.demo:
        print("Generating theta, h and theta*h distribution histograms...")
        
        plot_types = ['thetaxhx', 'hx', 'thetax']
        for plot_type in plot_types:
            values = expl.plot_distribution_h(test_loader, plot_type=plot_type)
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

            print('len values', len(values))
            plt.hist(values, color = plot_color, edgecolor = '#CCE6FF', bins=20)
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            plt.savefig(results_path + '/histogram' + plot_type + '.png', format = "png", dpi=300)
if __name__ == "__main__":
    main()
