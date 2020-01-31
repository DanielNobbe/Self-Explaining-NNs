# Revisiting Self-Explaining Neural Networks
This folder contains all code used to generate the results used in the paper "Revisiting Self-Explaining Neural Networks", by J. Goedhart, L. Jansen, H. Lim, and D. Nobbe.

# Prerequisites
1. Clone or download this repository to a local disk. It includes the MNIST and COMPAS datasets, so that will take a few minutes. 
2. Create a new conda environment (from a terminal in this folder) with the following command:
conda create -f SENN_environment.yml --name RSENN


# Notebook
With the notebook, all results can be generated. Do not run all cells at once, but only select the cells required, as some of them will trigger a lengthy training process. Pre-trained models are included in the repository.

# Main file
The main.py file in this folder can also be used for generating the same results. Run "python main.py --help" to see the possible flags.
Some extra information:
mnist5: Model with 5 concepts that can be used on the MNIST dataset.
mnist22: Model with 22 concepts that can be used on the MNIST dataset. We refer to our paper for more information on concepts.
Demo mode: Generates all plots for the specified model, with the exception of the faithfulness box plots.
noplot mode: Generates the faithfulness box plots.

