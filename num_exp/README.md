# Numerical Experiments

In this directory we keep all the files needed to reproduce all results reported in our manuscript. 

## Running our numerical experiments

The code contains the following files to reproduce the results of our manuscript:

1. For each method, RUG, FairRUG, FSDT, BinOCT, CG, and FairCG, we have a script file to initialize the k-fold cross validation. These script files call functions inside the helper files to execute the cross validtion. The script files are called `RUG_cv.py`, `FairRUG_cv.py`, `FSDT_cv.py`, `BinOCT_cv.py`, `CG_cv.py`, and `FairCG_cv.py`, respectively. Those are the files to run for the cross validation. Any parameters for the hyperparameter tuning can be adapted in those files.

2. Helper files.
- The script `grid_search_helpers.py` is used to run the k-fold cross validation for RUG, FairRUG, FSDT, BinOCT, CG, and FairCG. This script ensures that regardless of which method used, the way the data is handled is the same and hence results of all methods are comparable. Because some of the other methods cannot be used with sklearn's implementation of cross validation, we wrote this file.
- `FSDT_helpers.py` -- helper functions for FSDT, primarily to calculate performance metrics reported in the manuscript
- `CG_helpers.py` -- helper functions for CG and FairCG

3. The folder `case_study` includes the files to obtain the results reported in the case study of the manuscript (see section 4.3 and Appendix D).
- `case_study_helpers.py` -- helper functions to compute some performance metrics
- `CaseStudy.ipynb` -- Jupyter notebook to run to obtain the results 

4. Results. We have several folders for the results -- one for each method, called `results_w_RUG_manual`, `results_w_FairRUG_manual`, `results_w_FSDT_manual`, `results_w_BinOCT_manual`, `results_w_CG_manual`, and `results_w_FairCG_manual`. Each folder holds .txt files reporting the results for each dataset.

## Solvers

**Note that the default for `solver` option is 'gurobi'.** To use the Gurobi solver, you need to first install
it. The solver is freely available for academic use. Check the [related page](https://www.gurobi.com/academia/academic-program-and-licenses/)
on Gurobi's website. The current version of our code also supports the open source solver [GLPK](https://www.gnu.org/software/glpk/) (set `solver='glpk'`).
