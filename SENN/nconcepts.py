import matplotlib.pyplot as plt

nconcepts = range(2, 11)
prec_1 = [98.180, 98.010, 98.160, 98.270, 98.070, 98.080, 98.170, 98.220, 98.170]
prec_5 = [99.970, 99.990, 99.970, 99.990, 99.960, 99.990, 100.000, 99.980, 99.980]

plt.plot(nconcepts, prec_1, label = 'prec_1')
plt.plot(nconcepts, prec_5, label = 'prec_5')
plt.show()