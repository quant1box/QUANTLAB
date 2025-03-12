
#%%
from typing import List
import QUANTAXIS as QA
from datetime import date, datetime
import numpy as np
import pandas as pd
# import graphviz
from scipy.stats import rankdata
# from factors.func import*

from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from alphalens import utils
from alphalens import tears

#%%
# -------------------------------------------------------------------

x0 = np.arange(-1, 1, 1/10.)
x1 = np.arange(-1, 1, 1/10.)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1,
                       color='green', alpha=0.5)
plt.show()


# -------------------------------------------------------------------





# -------------------------------------------------------------------
