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
from SENN.eval_utils import sample_local_lipschitz

from SENN.models import GSENN
from SENN.conceptizers import input_conceptizer, image_fcc_conceptizer
from SENN.parametrizers import  dfc_parametrizer
from SENN.aggregators import additive_scalar_aggregator
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer
import itertools

from robust_interpret.explainers import gsenn_wrapper
from robust_interpret.utils import lipschitz_feature_argmax_plot

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

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
            theta = self(x, y = pred_class).data.numpy().squeeze()

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
            theta = self(x, y = pred_class).data.numpy().squeeze()

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
        # First, turn inputs into concepts
        # if not inputs_are_concepts:
        #     # x = x.type(torch.FloatTensor)
        #     x = x.unsqueeze(dim = 0)
        #     h_x = self.net.forward(x, h_options = -1)
        #     # print("h_x: ",h_x)
        #     f = self.net.forward(x, h_x = h_x, h_options = 1)
        # # Then, use concepts to forward pass through the model
        # # if inputs_are_concepts:
        # else:
        f   = self.net.forward(x.reshape(1,-1)) # model is compute_proba function, not neural model - we need to add output layer to compute probabilities. Not necessary for now TODO
        h_x = x
        # So for now, we use self.net
        # else:
        #     f = self.model.forward(x.reshape(1,-1), h_x = h_x,  h_options = 1)
        pred_class = f.argmax()
        # else:
        #     pred_class = int(target.item()) # Add this to MNIST implementation
        attributions = self(x, y = target) # attributions are theta values (i think)
        deltas = []
        for i in range(h_x.shape[0]):
            x_p = x.clone()
            x_p[i] = reference_value # uncomment this to be compatible with uci dataset #TODO: Make compatible with both compas and mnist
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
            # if not os.path.isdir(save_path_or):
            #     os.mkdir(save_path_or)
            save_path_alt = save_path + 'alternative'
            # if not os.path.isdir(save_path_alt):
            #     os.mkdir(save_path_alt)
            ex_att = max( max(max(attributions_plot) , max(attributions_plot_alt)), 
            -min(min(attributions_plot) , min(attributions_plot_alt)))
            lim_prob_drop = max(max(prob_drops), -min(prob_drops))
            limits = [ex_att, lim_prob_drop]
            plot_prob_drop(attributions_plot.squeeze(), prob_drops, save_path = save_path_or, limits=limits) # remove [0] after attributions for uci
            if alternative:
                plot_prob_drop(attributions_plot_alt.squeeze(), prob_drops, save_path = save_path_alt, limits=limits)
        return prob_drops, attributions, attributions_plot_alt

    # def compute_dependencies(self, x, reference_value = 0, plot = False, save_path = None, inputs_are_concepts = False):
    #     if not inputs_are_concepts:
    #             # x = x.type(torch.FloatTensor)
    #             x = x.unsqueeze(dim = 0)
    #             h_x = self.net.forward(x, h_options = -1)
    #             f = self.net.forward(x, h_x = h_x, h_options = 1)
    #         # Then, use concepts to forward pass through the model
    #         # if inputs_are_concepts:
    #     else:
    #         h_x = x
    #         # x = h_x # model is compute_proba function, not neural model - we need to add output layer to compute probabilities. Not necessary for now TODO
    #         f   = self.net.forward(x.reshape(1,-1))
    #     thetas = self(x) # attributions are theta values (i think)
    #     dependencies = thetas.squeeze() * h_x.cpu().detach().numpy().squeeze()
    #     return dependencies, thetas


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
                #print(scores.mean(), len(group))
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)

def load_compas_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size=64):
    df= pd.read_csv("data/propublica_data_for_fairml.csv")
    # Binarize num of priors var? Or normalize it 0,1?
    df['Number_of_Priors'] = np.sqrt(df['Number_of_Priors'])/(np.sqrt(38))
    compas_rating = df.score_factor.values # This is the target??
    df = df.drop("score_factor", 1)

    pruned_df, pruned_rating = find_conflicting(df, compas_rating)
    x_train, x_test, y_train, y_test   = train_test_split(pruned_df, pruned_rating, test_size=0.1, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)

    feat_names = list(x_train.columns)
    print("feature names", feat_names)
    x_train = x_train.values # pandas -> np
    x_test  = x_test.values

    Tds = []
    Loaders = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler(with_std=False, with_mean=False) # DOn't scale to make consitient with LIME/SHAP script
        transformed = scaler.fit_transform(foldx)
        #transformed = foldx
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

    model        = GSENN(conceptizer, parametrizer, aggregator)#, learn_h = args.train_h)

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

    # train_acc = trainer.validate(train_loader, fold = 'train')
    # valid_acc = trainer.validate(valid_loader, fold = 'valid')
    # test_acc = trainer.validate(test_loader, fold = 'test')

    # results['train_accuracy'] = train_acc
    # results['valid_accuracy']  = valid_acc
    # results['test_accuracy']  = test_acc
    # print('Train accuracy: {:8.2f}'.format(train_acc))
    # print('Valid accuracy: {:8.2f}'.format(valid_acc))
    # print('Test accuracy: {:8.2f}'.format(test_acc))


    #noise_stability_plots(model, test_tds, cuda = args.cuda, save_path = results_path)

    # lips, argmaxes = sample_local_lipschitz(model, test, mode = 2, top_k = 10, max_distance = 3)

    # max_lip = lips.max()
    # imax = np.unravel_index(np.argmax(lips), lips.shape)[0]
    # jmax = argmaxes[imax][0][0]
    # print('Max Lip value: {}, attained for pair ({},{})'.format(max_lip, imax, jmax))

    # x      = test.tensors[0][imax]
    # argmax = test.tensors[0][jmax]

    # pred_x = model(Variable(x.view(1,-1), volatile = True)).data
    # att_x = model.thetas.data.squeeze().numpy().squeeze()

    # pred_argmax = model(Variable(argmax.view(1,-1), volatile = True)).data
    # att_argmax = model.thetas.data.squeeze().numpy().squeeze()

    # pdb.set_trace()
    # results['x_max']      = x
    # results['x_argmax']      = argmax
    # results['test_discrete_glip']      = lips
    # results['test_discrete_glip_argmaxes'] = argmaxes


    # print('Local g-Lipschitz estimate: {:8.2f}'.format(lips.mean()))

    # fpath = os.path.join(results_path, 'discrete_lip_gsenn')
    # ppath = os.path.join(results_path, 'relevance_argmax_gsenn')

    # pickle.dump(results,  open(fpath+'.pkl',"wb")) # FOrmerly model_metrics
    #
    # print(ppath)
    # lipschitz_feature_argmax_plot(x, argmax, att_x, att_argmax,
    #                               feat_names = feat_names,
    #                               digits=2, figsize=(5,6), widths=(2,3),
    #                               save_path=ppath + '.pdf')
    # Added by Lennert:
    # set up input arguments for GSENN wrapper
    features = None
    classes = [0, 1]
    model.eval()

    #
    expl = new_wrapper(model,
                        mode = 'classification',
                        input_type = 'feature',
                        feature_names = 'features',
                        class_names = 'classes',
                        train_data = train_loader,
                        skip_bias = True,
                        verbose = False)

    ### Consistency analysis

    correlations = np.array([])
    altcorrelations = np.array([])
    for i, (inputs, targets) in enumerate(tqdm(test_loader)):
            # get the inputs
            # if model.cuda:
            #     inputs, targets = inputs.cuda(), targets.cuda()
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
            # print("i: ", i)

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
    plt.savefig(results_path + '/faithfulness_box_plot.png', format = "png", dpi=300)


    # expl = gsenn_wrapper(model,
    #                     mode = 'classification',
    #                     input_type = 'feature',
    #                     feature_names = 'features',
    #                     class_names = 'classes',
    #                     train_data = train_loader,
    #                     skip_bias = True,
    #                     verbose = False)

    # Iteratively perform faithfulness analysis on test inputs
    # correlations = np.array([])
    # altcorrelations = np.array([])
    # for i, (inputs, targets) in enumerate(test_loader):
    #         # get the inputs
    #         # if model.cuda:
    #         #     inputs, targets = inputs.cuda(), targets.cuda()
    #         input_var = torch.autograd.Variable(inputs, volatile=True)
    #         target_var = torch.autograd.Variable(targets)
    #         save_path = results_path + '/faithfulness' + str(i) + '/'
    #         if not os.path.isdir(save_path):
    #             os.mkdir(save_path)
    #         corrs, altcorrs = expl.compute_dataset_consistency(input_var, targets = target_var, inputs_are_concepts = False, save_path = save_path)
    #         correlations = np.append(correlations, corrs)
    #         altcorrelations = np.append(altcorrelations, altcorrs)
    #         print("i: ", i)
    #         # if i > 0:
    #         #     print("Breaking out of loop")
    #         #     break
    # average_correlation = np.sum(correlations)/len(correlations)
    # std_correlation = np.std(correlations)
    # average_alt_correlation = np.sum(altcorrelations)/len(altcorrelations)
    # std_alt_correlation = np.std(altcorrelations)
    # print("Average correlation:", average_correlation)
    # print("Standard deviation of correlations: ", std_correlation)
    # print("Average alternative correlation:", average_alt_correlation)
    # print("Standard deviation of alternative correlations: ", std_alt_correlation)

    ## JOOSJE: STABILITIES
    # stabilities_1, stabilities_2 = expl.compute_stability(test_loader)
    # stabilities = [stabilities_1, stabilities_2]

    # box = plt.boxplot(stabilities, patch_artist=True, labels=['theta(x)', 'theta(x) h(x)'])
    # colors = ['blue', 'purple']
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_facecolor(color)
    # plt.show()
    if args.demo:
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
