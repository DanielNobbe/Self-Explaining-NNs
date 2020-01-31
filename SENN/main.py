import os
import argparse

parser = argparse.ArgumentParser(description='Self-Explaining Neural Net Classifier')

parser.add_argument('--data', default=False, help='Dataset/model to use: compas, mnist5 or mnist22')
parser.add_argument('--train', action = 'store_true', default=False, help='Include if training needs to be done on selected model')
parser.add_argument('--demo', action = 'store_true', default = False, help='Include if demo mode needs to be used')
parser.add_argument('--noplot', action = 'store_true', default = False, help='Include if noplot mode needs to be used')

args = parser.parse_args()

if args.data == 'mnist5':
    if args.train:
        os.system('python scripts/main_mnist.py --nconcepts 5 --train')
    elif args.demo:
        os.system('python scripts/main_mnist.py --nconcepts 5 --demo')
    elif args.noplot:
        os.system('python scripts/main_mnist.py --nconcepts 5 --noplot')
    else:
        print("Do not use multiple flags! Ending..")
elif args.data == 'mnist22':
    if args.train:
        os.system('python scripts/main_mnist.py --nconcepts 22 --train')
    elif args.demo:
        os.system('python scripts/main_mnist.py --nconcepts 22 --demo')
    elif args.noplot:
        os.system('python scripts/main_mnist.py --nconcepts 22 --noplot')
    else:
        print("Do not use multiple flags! Ending..")
elif args.data == 'compas':
    if args.train:
        os.system('python scripts/main_mnist.py --h_type input --train')
    elif args.demo:
        os.system('python scripts/main_mnist.py --h_type input --demo')
    else:
        print("Do not use multiple flags! Ending..")
