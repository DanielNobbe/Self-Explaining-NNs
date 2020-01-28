import os, sys

for i in range(2,11):
	print(i)
	os.system('cmd /k python scripts/main_mnist.py --nconcepts ' + str(i))
