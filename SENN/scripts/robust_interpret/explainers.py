# -*- coding: utf-8 -*-
""" Code robustness evaluation in interpretability methods. Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

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

import os
from pprint import pprint
import scipy as sp
import numpy as np
from skopt import gp_minimize, gbrt_minimize
from functools import partial
from collections import defaultdict
from itertools import chain
import pickle
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
#import multiprocessing
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances

# UTILS
from .utils import deepexplain_plot
from .utils import rgb2gray_converter
from .utils import topk_argmax

import pdb


#   - make all wrappers uniform by passing model, not just predict_proba method,
#     since DeepX needs the full model. Can then take whatever method is required inside each class.
#   - seems like for deepexplain can choose between true and predicted class easlity. Add that option.

try:
    from SENN.utils import plot_prob_drop
except:
    print('Couldnt find SENN')

# def test():
#     print("5")

# def make_keras_picklable():
#     def __getstate__(self):
#         model_str = ""
#         with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
#             keras.models.save_model(self, fd.name, overwrite=True)
#             model_str = fd.read()
#         d = {'model_str': model_str}
#         return d

#     def __setstate__(self, state):
#         with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
#             fd.write(state['model_str'])
#             fd.flush()
#             model = keras.models.load_model(fd.name)

#         self.__dict__ = model.__dict__

#     cls = keras.models.Model
#     cls.__getstate__ = __getstate__
#     cls.__setstate__ = __setstate__


# def _parallel_lipschitz(wrapper, i, x, bound_type, eps, n_calls):
#     # make_keras_picklable()
#     print('\n\n ***** PARALLEL : Example ' + str(i) + '********')
#     print(wrapper.net.__dict__.keys())
#     if 'model' in wrapper.net.__dict__.keys():
#         l, _ = wrapper.local_lipschitz_estimate(
#             x, eps=eps, bound_type=bound_type, n_calls=n_calls)
#     else:
#         l = None
#     return l


class explainer_wrapper(object):
    def __init__(self, model, mode, explainer, multiclass=False,
                 feature_names=None, class_names=None, train_data=None):
        self.mode = mode
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.multiclass = multiclass
        self.class_names = class_names
        self.train_data = train_data  # Necessary only to get data distrib stats

        if self.train_data is not None:
            # These are per-feature dim statistics
            print("Computing train data stats...")
            self.train_stats = {
                'min': self.train_data.min(0),
                'max': self.train_data.max(0),
                'mean': self.train_data.mean(0),
                'std': self.train_data.std(0)
            }
            # pprint(self.train_stats)

    # def estimate_dataset_lipschitz(self, dataset, continuous=True, eps=1, maxpoints=None,
    #                                optim='gp', bound_type='box', n_jobs=1, n_calls=10, verbose=False):
    #     """
    #         Continuous and discrete space version.

    #     """
    #     # make_keras_picklable()
    #     n = len(dataset)
    #     if maxpoints and n > maxpoints:
    #         dataset_filt = dataset[np.random.choice(n, maxpoints)]
    #     else:
    #         dataset_filt = dataset[:]
    #     if n_jobs > 1:
    #         Lips = Parallel(n_jobs=n_jobs, max_nbytes=1e6, verbose=verbose)(delayed(_parallel_lipschitz)(
    #             self, i=i, x=x, bound_type=bound_type, eps=eps, n_calls=n_calls) for i, x in enumerate(dataset_filt))
    #     else:
    #         Lips = []
    #         for x in dataset_filt:
    #             l, _ = self.local_lipschitz_estimate(x, optim=optim,
    #                                                  bound_type=bound_type, eps=eps, n_calls=n_calls, verbose=verbose)
    #             Lips.append(l)
    #     print(
    #         'Missed points: {}/{}'.format(sum(x is None for x in Lips), len(dataset_filt)))
    #     Lips = np.array([l for l in Lips if l])
    #     return Lips

    # # def lipschitz_ratio(self, x=None, y=None, reshape=None, minus=False):
    # #     """
    # #         If minus = True, returns minus this quantitiy.

    # #         || f(x) - f(y) ||/||x - y||

    # #     """
    # #     # NEed this ungly hack because skopt sends lists
    # #     if type(x) is list:
    # #         x = np.array(x)
    # #     if type(y) is list:
    # #         y = np.array(y)
    # #     if reshape is not None:
    # #         # Necessary because gpopt requires to flatten things, need to restrore expected sshape here
    # #         x = x.reshape(reshape)
    # #         y = y.reshape(reshape)
    # #     #print(x.shape, x.ndim)
    # #     multip = -1 if minus else 1
    # #     return multip * np.linalg.norm(self(x) - self(y)) / np.linalg.norm(x - y)

    # def local_lipschitz_estimate(self, x, optim='gp', eps=None, bound_type='box',
    #                              clip=True, n_calls=100, njobs = -1, verbose=False):
    #     """
    #         Compute one-sided lipschitz estimate for explainer. Adequate for local
    #          Lipschitz, for global must have the two sided version. This computes:

    #             max_z || f(x) - f(z)|| / || x - z||

    #         Instead of:

    #             max_z1,z2 || f(z1) - f(z2)|| / || z1 - z2||

    #         If eps provided, does local lipzshitz in:
    #             - box of width 2*eps along each dimension if bound_type = 'box'
    #             - box of width 2*eps*va, along each dimension if bound_type = 'box_norm' (i.e. normalize so that deviation is eps % in each dim )
    #             - box of width 2*eps*std along each dimension if bound_type = 'box_std'

    #         max_z || f(x) - f(z)|| / || x - z||   , with f = theta

    #         clip: clip bounds to within (min, max) of dataset

    #     """
    #     # Compute bounds for optimization
    #     if eps is None:
    #         # If want to find global lipzhitz ratio maximizer - search over "all space" - use max min bounds of dataset fold of interest
    #         # gp can't have lower bound equal upper bound - so move them slightly appart
    #         lwr = self.train_stats['min'].flatten() - 1e-6
    #         upr = self.train_stats['max'].flatten() + 1e-6
    #     elif bound_type == 'box':
    #         lwr = (x - eps).flatten()
    #         upr = (x + eps).flatten()
    #     elif bound_type == 'box_std':
    #         # gp can't have lower bound equal upper bound - so set min std to 0.001
    #         lwr = (
    #             x - eps * np.maximum(self.train_stats['std'], 0.001)).flatten()
    #         upr = (
    #             x + eps * np.maximum(self.train_stats['std'], 0.001)).flatten()
    #     if clip:
    #         lwr = lwr.clip(min=self.train_stats['min'].min())
    #         upr = upr.clip(max=self.train_stats['max'].max())
    #     bounds = list(zip(*[lwr, upr]))
    #     if x.ndim > 2:
    #         # This is an image, will need to reshape
    #         orig_shape = x.shape
    #         x = x.flatten()
    #     else:
    #         orig_shape = x.shape

    #     # Run optimization
    #     if optim == 'gp':
    #         print('Running BlackBox Minimization with Bayesian Optimization')
    #         # Need minus because gp only has minimize method
    #         f = partial(self.lipschitz_ratio, x,
    #                     reshape=orig_shape, minus=True)
    #         res = gp_minimize(f, bounds, n_calls=n_calls,
    #                           verbose=verbose, n_jobs=njobs)
    #     elif optim == 'gbrt':
    #         print('Running BlackBox Minimization with Gradient Boosted Trees')
    #         f = partial(self.lipschitz_ratio, x,
    #                     reshape=orig_shape, minus=True)
    #         res = gbrt_minimize(f, bounds, n_calls=n_calls,
    #                             verbose=verbose, n_jobs=njobs)

    #     lip, x_opt = -res['fun'], np.array(res['x'])
    #     if verbose:
    #         print(lip, np.linalg.norm(x - x_opt))
    #     return lip, x_opt.reshape(orig_shape)

    # def estimate_discrete_dataset_lipschitz(self, dataset, eps = None, top_k = 1,
    #     metric = 'euclidean', same_class = False):
    #     """
    #         For every point in dataset, find pair point y in dataset that maximizes
    #         Lipschitz: || f(x) - f(y) ||/||x - y||

    #         Args:
    #             - dataset: a tds obkect
    #             - top_k : how many to return
    #             - max_distance: maximum distance between points to consider (radius)
    #             - same_class: ignore cases where argmax g(x) != g(y), where g is the prediction model
    #     """
    #     Xs  = dataset
    #     n,d = Xs.shape
    #     Fs = self(Xs)
    #     Preds_prob = self.model(Xs)
    #     Preds_class = Preds_prob.argmax(axis=1)
    #     num_dists = pairwise_distances(Fs)#, metric = 'euclidean')
    #     den_dists = pairwise_distances(Xs, metric = metric) # Or chebyshev?
    #     if eps is not None:
    #         nonzero = np.sum((den_dists > eps))
    #         total   = den_dists.size
    #         print('Number of zero denom distances: {} ({:4.2f}%)'.format(
    #             total - nonzero, 100*(total-nonzero)/total))
    #         den_dists[den_dists > eps] = -1.0 #float('inf')
    #     # Same with self dists
    #     den_dists[den_dists==0] = -1 #float('inf')
    #     if same_class:

    #         for i in range(n):
    #             for j in range(n):
    #                 if Preds_class[i] != Preds_class[j]:
    #                     den_dists[i,j] = -1

    #     ratios = (num_dists/den_dists)
    #     argmaxes = {k: [] for k in range(n)}
    #     vals, inds = topk_argmax(ratios, top_k)
    #     argmaxes = {i:  [(j,v) for (j,v) in zip(inds[i,:],vals[i,:])] for i in range(n)}
    #     return vals.squeeze(), argmaxes

    def compute_dataset_consistency(self,  dataset, reference_value = 0, save_path = None):
        """
            does compute_prob_drop for all dataset, returns stats and plots

        """
        drops = []
        atts  = []
        corrs = []
        for x in dataset:
            p_d, att = self.compute_prob_drop(x, save_path = save_path)
            p_d = p_d.squeeze()
            att = att.squeeze()
            drops.append(p_d)
            atts.append(att)
            #pdb.set_trace()
            #assert len(p_d).shape[0] == atts.shape[0], "Attributions has wrong size"
            #pdb.set_trace()
            corrs.append(np.corrcoef(p_d, att)[0,1]) # Compute correlation per sample, then aggreate

        corrs = np.array(corrs)
        # pdb.set_trace()
        # drops = np.stack(drops)
        # atts  = np.stack(atts)
        #
        # np.corrcoef(drops.flatten(), atts.flatten())
        return corrs

    def compute_prob_drop(self, x, reference_value = 0, plot = True, save_path = None):
        """
            Warning: this only works for SENN if concepts are inputs!!!!
            In that case, must use the compute prob I have in SENN class.
        """
        f   = self.model(x.reshape(1,-1))
        pred_class = f.argmax()
        attributions = self(x)
        deltas = []
        for i in tqdm(range(x.shape[0])):
            x_p = x.clone()
            x_p[i] = reference_value
            f_p = self.model(x_p.reshape(1,-1))
            delta_i = (f - f_p)[0,pred_class]
            deltas.append(delta_i)
        prob_drops = np.array(deltas)
        if plot:
            plot_prob_drop(attributions[0], prob_drops, save_path = save_path)
        return prob_drops, attributions


    def plot_image_attributions_(self, x, attributions):
        """
            x: either (n x d) if features or (n x d1 x d2) if gray image or (n x d1 x d2 x c) if color
        """

        # for different types of data _display_attribs_image , etc
        n_cols = 4
        n_rows = max(int(len(attributions) / 2), 1)
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
        for i, a in enumerate(attributions):
            row, col = divmod(i, 2)
            if  (x.ndim == 3) or (x.shape[3] == 1):
                # Means single channel, like MNIST, plot in gray
                # I had this for mnist, worked:
                deepexplain_plot(x[i].reshape(
                    x.shape[1], x.shape[2]), cmap='Greys', axis=axes[row, col * 2]).set_title('Original')
                deepexplain_plot(a.reshape(
                    x.shape[1], x.shape[2]), xi=x[i], axis=axes[row, col * 2 + 1]).set_title('Attributions')
            else:
                ax = axes[row, col * 2]
                xi = x[i]
                xi = (xi - np.min(xi))
                xi /= np.max(xi)
                ax.imshow(xi)
                ax.set_title('Original')
                ax.axis('off')
                deepexplain_plot(a, xi=x[i], axis=axes[row, col * 2 + 1],
                                 dilation=.5, percentile=99, alpha=.2).set_title('Attributions')
                #deepexplain_plot(a, xi = x[i], axis=axes[row,col*2+1],dilation=.5, percentile=99, alpha=.2).set_title('Attributions')
        plt.show()


class gsenn_wrapper(explainer_wrapper):
    """
        Wrapper around gsenn
    """
    # def __init__(self, keras_wrapper, explainer, multiclass, feat_names, class_to_explain, train_data, nsamples = 100):
    #     super().__init__(keras_wrapper.predict, None, multiclass, feat_names, train_data)
    def __init__(self, model, mode,input_type, multiclass=False, feature_names=None,
                 class_names=None, train_data=None, num_features=None, categorical_features=None,
                 nsamples=100, skip_bias = True, verbose=False):

        self.input_type = input_type
        self.net = model
        self.skip_bias = skip_bias # If we added bias term in SENN, remove from attributions

        super().__init__(self.net.predict_proba, mode, None,
                         multiclass, feature_names, class_names, None)


        stack = []
        for input,_ in train_data:
            stack.append(input.squeeze().numpy())

        transformed_dataset = np.concatenate(stack)
        if train_data is not None:
            print("Computing train data stats...")
            self.train_stats = {
                'min': np.min(transformed_dataset,0),
                'max': np.max(transformed_dataset,0),
                'mean': np.mean(transformed_dataset,0),
                'std': np.std(transformed_dataset,0),
            }

        self.verbose = verbose


    def __call__(self, x, y=None, return_dict=False, x_raw=None, show_plot=False): # Aangepast: return_dict=False
        """
            x_raw: if provided, will plot this instead of x (useful for images that have been processed)

        """
        import torch
        from torch import Tensor, from_numpy
        from torch.autograd import Variable

        if type(x) is np.ndarray:
            x_t = from_numpy(x).float()
        elif type(x) is Tensor:
            x_t = x.clone()
        else:
            print(type(x))
            raise ValueError("Unrecognized data type")

        if x_t.dim() == 1:
            x_t = x_t.view(1,-1)
        elif x_t.dim() == 2 and self.input_type == 'image':
            # Single image, gray. Batchify with channels first
            x_t = x_t.view(1,1,x_t.shape[0],x_t.shape[1])
        elif x_t.dim() == 3:  # Means x is image and single example passed. Batchify.
            x_t = x_t.view(1, x_t.shape[0], x_t.shape[1], x_t.shape[2])

        if self.input_type == 'image' and (x_t.shape[1] in [1,3]):
            channels_first = True
        elif self.input_type == 'image':
            channels_first = False
            x_t = x_t.transpose(1,3).transpose(2,3)

        if self.multiclass:
            pass

        x_t = Variable(x_t, volatile = True)
        pred = self.net(x_t)

        attrib_mat = self.net.thetas.data.cpu()

        nx, natt, nclass = attrib_mat.shape

        if y is None:
            # Will compute attribution w.r.t. predicted class
            vals, argmaxs = torch.max(pred.data, -1)
            attributions = attrib_mat.gather(2,argmaxs.cpu().view(-1,1).unsqueeze(2).repeat(1,natt,nclass))[:,:,0].numpy()

        else:
            # Will compute attribution w.r.t. to provided classes
            if y.type() != torch.LongTensor:
                y = y.type(torch.LongTensor)
            if y.nelement()>1:
                _, expl_target_class = torch.max(y)
            else:
                expl_target_class = y


            if self.input_type == 'feature':
                attributions = attrib_mat.cpu().view(1,-1).numpy()
            else:
                attributions = attrib_mat.gather(2,expl_target_class.cpu().view(-1,1).unsqueeze(2).repeat(1,natt,nclass))[:,:,0].numpy()

        if self.skip_bias and getattr(self.net.conceptizer, "add_bias", None):
            attributions = attributions[...,:-1]

        if show_plot:
            if self.input_type == 'image':
                x_plot = x_raw if (x_raw is not None) else x
                if x_raw is None and channels_first:
                    x_plot = x_plot.transpose(1,3).transpose(1,2)
                if not type(x_plot) is np.ndarray:
                    x_plot = x_plot.numpy().squeeze()

                self.plot_image_attributions_(x_plot, attributions)
            else: 
                exp_dict_y = dict(zip(feat_names, att_y))
                _ = plot_dependencies(exp_dict_y, x = y, sort_rows = False, scale_values = False, ax = ax[1,1],
                                        show_table = True, digits = digits, ax_table = ax[1,0], title = 'Explanation')

        self.explanation = attributions
        vals = attributions.reshape(attributions.shape[0], -1)

        if not return_dict:
            return vals
        else:
            exp_dict = dict(zip(self.feature_names, vals))
            return exp_dict
