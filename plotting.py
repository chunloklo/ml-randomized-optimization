import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

############## Options to generate nice figures
fig_width_pt = 500.0  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean * 2/3  # height in inches
fig_size = [fig_width, fig_height]

############## Colors I like to use
my_yellow = [235. / 255, 164. / 255, 17. / 255]
my_blue = [58. / 255, 93. / 255, 163. / 255]
dark_gray = [68./255, 84. /255, 106./255]
my_red = [163. / 255, 93. / 255, 58. / 255]

my_color = dark_gray # pick color for theme

params_keynote = {
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}
############## Parameters I use for IEEE papers
params_ieee = {
    'figure.autolayout' : True,
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}

############## Choose parameters you like
matplotlib.rcParams.update(params_ieee)
plt.rcParams.update({'legend.fontsize': 5.6})


import numpy as np
#RHC
# test_acc = np.load("npData/20190302-175041-RHC-NN-score-0.1s-40-10r.npy")
# valid_acc = np.load("npData/20190302-175041-RHC-NN-valid_score-0.1s--40-10r.npy")
# vals = range(len(test_acc))

# test_acc2 = np.load("npData/20190302-175402-RHC-NN-score-0.3s-40-10r.npy")
# valid_acc2 = np.load("npData/20190302-175402-RHC-NN-valid_score-0.3s--40-10r.npy")
# vals2 = range(len(test_acc2))

#SA
# test_acc = np.load("npData/20190302-182640-SA-NN-train-geo-4000i-40a.npy")
# valid_acc = np.load("npData/20190302-182640-SA-NN-valid-geo-4000i-40a.npy")
# vals = range(len(test_acc))

# test_acc2 = np.load("npData/20190302-185749-SA-NN-train-arith-4000i-40a.npy")
# valid_acc2 = np.load("npData/20190302-185749-SA-NN-valid-arith-4000i-40a.npy")
# vals2 = range(len(test_acc2))

#GA
s1 = "20190302-194219-GA-NN-test-400p-10a-100i.npy"
s2 = "20190302-194219-GA-NN-valid-400p-10a-100i.npy"

test_acc = np.load("npData/" + s1)
valid_acc = np.load("npData/" + s2)
vals = range(len(test_acc))

s1 = "20190302-194230-GA-NN-test-800-10a-100i.npy"
s2 = "20190302-194230-GA-NN-valid-800-10a-100i.npy"


test_acc2 = np.load("npData/" + s1)
valid_acc2 = np.load("npData/" + s2)
vals2 = range(len(test_acc2))


ax = plt.subplot(121)
ax.set_title("Geometric Decay")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(vals, test_acc, color=my_yellow, label='Train')
plt.plot(vals, valid_acc, color=my_blue, label='Validation')
plt.legend()

ax = plt.subplot(122)
ax.set_title("Arithmetic Decay")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.plot(vals2, test_acc2, color=my_yellow, label='Train')
plt.plot(vals2, valid_acc2, color=my_blue, label='Validation')
plt.legend()



print(np.max(test_acc))
print(np.max(test_acc2))
plt.savefig("../tex/figures/GA-NN.pdf")
plt.show()
