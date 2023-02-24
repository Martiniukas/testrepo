from pyspark.sql import SparkSession
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
import pyspark as spark


# Define parameters

df = pd.read_csv('malloc_df.csv')

numpy_array = df.to_numpy()
r_nom = numpy_array[:, 2]
r_dev = numpy_array[:, 3]
v_nom = numpy_array[:, 4]
v_dev = numpy_array[:, 5]

r_nom = np.reshape(r_nom, (len(r_nom),1))
r_dev = np.reshape(r_dev, (len(r_dev),1))
v_nom = np.reshape(v_nom, (len(v_nom),1))
v_dev = np.reshape(v_dev, (len(v_dev),1))

no_articles = 13
s_0 = 34810.0

# Produce A matrix
dups_article = df.pivot_table(columns=['article'], aggfunc='size')

branchcount = dups_article.to_numpy()
blocks = [np.ones([1,branchcount[x]],'int') for x in range(len(branchcount))]
A = scipy.sparse.block_diag(blocks)


r_hat = scipy.sparse.diags(r_dev.T, [0])
v_hat = scipy.sparse.diags(v_dev.T, [0])
#sum_branches = sum(branchcount)
sum_branches = len(r_dev)
q = 0.8
e = np.ones((sum_branches,1))
ones_n = np.ones((no_articles,1))
zeros = np.zeros((sum_branches,1))
Gamma_r = 0
Gamma_v = 6

# Define variables
t = cp.Variable(1)
x = cp.Variable((sum_branches,1), boolean=True)
y = cp.Variable((sum_branches,1))

p_r = cp.Variable((sum_branches,1))
z_r = cp.Variable(1)

p_v = cp.Variable((sum_branches,1))
z_v = cp.Variable(1)

# Define objective
objective = cp.Maximize(t)

constraint = [t - r_nom.T@x + z_r*Gamma_r + e.T@p_r <= 0,
              -v_nom.T@x + z_v*Gamma_v + e.T@p_v <=-q*s_0,
              z_r*e + p_r >= r_hat@x,
              z_v*e + p_v >= v_hat@x,
              p_v>=zeros,
              p_r>=zeros,
              z_r>=zeros,
              z_v>=zeros,
              A@x == ones_n
             ]

# Solve!
problem = cp.Problem(objective, constraint)
solution = problem.solve(solver=cp.GUROBI, verbose=True)

### Calculates stuff
x_diag = scipy.sparse.diags(x.value.T, [0])

# Revenue
rev_unc= np.zeros(np.shape(p_r.value))
z_r_vec = np.zeros(np.shape(p_r.value))

if Gamma_r == 1:
    rev_unc[np.nonzero(x.value)[0][np.argmax(np.nonzero(x.value)[0])]] = -max(r_dev[np.nonzero(x.value)])

else:
    z_r_vec[np.nonzero(p_r.value)] = z_r.value/Gamma_r
    rev_unc[np.nonzero(p_r.value)[0]] = -r_dev[np.nonzero(p_r.value)[0]]
    #rev_unc =  x_diag@(-z_r_vec*Gamma_r*e - p_r.value)  # Uncertainty

rev = x_diag@(r_nom + rev_unc) # Total revenue with and w/o uncertainty



# Sales volume
v_unc = np.zeros(np.shape(p_r.value))
v_unc[np.nonzero(p_r.value)[0]] = v_dev[np.nonzero(p_r.value)[0]]
v = x_diag@(v_nom - v_unc)


print(f"Total revenue of the sales period for {no_articles} articles is {np.round(problem.value)} RUB")
print(f"Total initial inventory of the sales period is {round(s_0)} pieces")
print(f"Total final sales of the sales period will be {np.floor(sum(v)[0])} pieces")
print(f"This results in a sell through at {round(100*(sum(v)[0])/s_0)}%")