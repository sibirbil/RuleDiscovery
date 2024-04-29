# Numerical Experiments

In this directory we keep all the files needed to reproduce all results reported in our manuscript. 

## Running our numerical experiments

The code contains the following files to reproduce the results of our manuscript:

1. `run_exp.py` is a script file used to initialize the k-fold cross validation for all methods. This script file calls functions inside the helper files to execute the cross validtion. Any parameters for the hyperparameter tuning can be adapted in those files.

2. Helper files in `num_exp/helpers/`.
- The script `grid_search_helpers.py` is used to run the k-fold cross validation for RUG, FairRUG, FSDT, BinOCT, CG, and FairCG. This script ensures that regardless of which method used, the way the data is handled the same way and hence results of all methods are comparable. Because some of the other methods cannot be used with sklearn's implementation of cross validation, we wrote this file.
- `FSDT_helpers.py` -- helper functions for FSDT, primarily to calculate performance metrics reported in the manuscript
- `CG_helpers.py` -- helper functions for CG and FairCG

3. The file `CaseStudy.ipynb` is a Jupyter notebook used to obtain results reported in the case study of the manuscript.

4. Results. We have several folders for the results -- one for each method, called `results_w_RUG`, `results_w_FairRUG`, `results_w_FSDT`, `results_w_BinOCT`, `results_w_CG`, and `results_w_FairCG`. Each folder holds .txt files reporting the results for each dataset.

## Solvers

**Note that the default for `solver` option is 'gurobi'.** To use the Gurobi solver, you need to first install
it. The solver is freely available for academic use. Check the [related page](https://www.gurobi.com/academia/academic-program-and-licenses/)
on Gurobi's website. The current version of our code also supports the open source solver [GLPK](https://www.gnu.org/software/glpk/) (set `solver='glpk'`).
