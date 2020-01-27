
import pickle
from matplotlib import pyplot as plt
import numpy as np
from itertools import chain
import os
import torch

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

from random import sample
from collections import defaultdict

from SENN.eval_utils import estimate_dataset_lipschitz

from robust_interpret.explainers import gsenn_wrapper
from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot, lipschitz_feature_argmax_plot


def eval_stability(test_tds, model, scale = 0.5):

    distances = defaultdict(list)

    for i in tqdm(range(1000)):

	    x = Variable(test_tds[i][0].view(1,1,28,28), volatile = True)

	    true_class = test_tds[i][1][0].item()

	    pred = model(x)

	    theta = model.thetas.data.cpu().numpy().squeeze()

	    klass = pred.data.max(1)[1]
	    deps = theta[:,klass].squeeze()

	    # print("prediction", klass)
	    # print("dependencies", deps)

	    # Add noise to sample and repeat
	    noise = Variable(scale*torch.randn(x.size()), volatile = True)

	    pred = model(noise)

	    theta = model.thetas.data.cpu().numpy().squeeze()

	    klass_noise = pred.data.max(1)[1]
	    deps_noise = theta[:,klass].squeeze()

	    dist = np.linalg.norm(deps - deps_noise) / np.sqrt(len(theta))

	    distances[true_class].append(dist)

    return distances

def main():

    path = os.path.join(os.getcwd(), 'out/mnist/eval_stability') #"C:/Users/joosj/Documents/Master AI/FACT/FACT/SENN/out/mnist/eval_stability"
    models = os.listdir(path)

    # Load test data
    with open(os.path.join('out/mnist/eval_stability', 'test_tds.pkl'), "rb") as handle:
    	test_tds = pickle.load(handle)
    	handle.close()

    means = []
    stds = []
    concepts = []

    for name in models:
    	if '.py' not in name and '.pkl' not in name:
    		print(name)

    		concepts.append(int(name.split('Cpts')[1].split('_')[0]))

    		# Load Model
    		checkpoint = torch.load(os.path.join('out', 'mnist', 'eval_stability', name, 'model_best.pth.tar'), map_location=lambda storage, loc: storage)
    		model = checkpoint['model']

    		distances = eval_stability(test_tds, model, scale=0.05)
    		labels, data = distances.keys(), distances.values()

    		mean = sum(list(chain.from_iterable(list(data)))) / 1000
    		std = np.std(list(chain.from_iterable(list(data))))

    		do_bb_stability = True
    		if do_bb_stability:
    			expl = gsenn_wrapper(model,
                        mode      = 'classification',
                        input_type = 'image',
                        multiclass=True,
                        feature_names = features,
                        class_names   = classes,
                        train_data      = train_loader,
                        skip_bias = True,
                        verbose = False)

    			mini_test = test_tds[0][:20][0].numpy()
    			print(mini_test)

    			# lips = exp.estimate_dataset_lipschitz(mini_test, n_jobs = -1, bound_type = 'box_std', eps = , optim = ,  n_calls = , verbose = 2)
    		# do_bb_stability = True # Aangepast, was: True
		    # if do_bb_stability:
		    #     print('**** Performing black-box lipschitz estimation over subset of dataset ****')
		    #     maxpoints = 20
		    #     #valid_loader 0 it's shuffled, so it's like doing random choice
		    #     mini_test = test_tds[0][:20] #next(iter(valid_loader))[0][:maxpoints].numpy()

		    #     print(mini_test)
		    #     # lips = expl.estimate_dataset_lipschitz(mini_test,
		        #                                    n_jobs=-1, bound_type='box_std',
		        #                                    eps=args.lip_eps, optim=args.optim,
		        #                                    n_calls=args.lip_calls, verbose=2  


    		means.append(mean)
    		stds.append(std)



    plt.errorbar(concepts, means, yerr=stds, fmt='o')
    plt.title("Stability for different number of concepts")
    plt.show()

if __name__ == '__main__':
    main()