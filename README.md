# Rule Generation for Classification: Scalability, Interpretability, and Fairness

**Adia Lumadjeng, Tabea Röber, M. Hakan Akyüz, and Ş. Ilker Birbil**


We introduce a new rule-based optimization method for classification with constraints. The proposed method
takes advantage of linear programming and column generation, and hence, is scalable to large datasets. Moreover, the method returns a set of rules along with their optimal weights indicating the importance of each rule for learning. Through assigning cost coefficients to the rules and introducing additional constraints, we show that one can also consider inter pretability and fairness of the results. We test the performance of the proposed method on a collection of datasets and present two case studies to elaborate its different aspects. Our results show that a good compromise between interpretability and fairness on the one side, and accuracy on the other side, can be obtained by the proposed rule-based learning method.

You can find the details of both algorithms in [our manuscript](https://arxiv.org/abs/2104.10751).

This [notebook](RuleDiscovery.ipynb) illustrates how to use RUX and RUG.

## Installation

 1. Install [Anaconda Distribution](https://www.anaconda.com/products/individual).

 2. Create a new environment and install the necessary packages:

 `conda create -n rulediscovery --channel=conda-forge python=3.8 numpy pandas scikit-learn cvxpy cvxopt gurobi`

 3. Activate the current environment:

 `conda activate rulediscovery`

 4. Check whether the installation works with either one of the files explained below.

---

## Running Our Numerical Experiments

The code contains the following files to reproduce the results of our manuscript:

1. The script `ruxg_testing.py` is used to test the RUX and RUG algorithms in single fold. One or all datasets are selected from the list, and the code will produce the results of RF, ADA and GB, along with RUX(RF), RUX(ADA), RUX(GB), and RUG.
 
2. **TO BE COMPLETED ...** The folder `CVResults` consists of three files `ruxg_cv_others.py`, `ruxg_cv_RUX.py` and `ruxg_cv_RUG.py`, to produce the cross validated results of Table EC.3 and Table EC.4 of our manuscript. Table EC.3 consists of the predictive performances of existing methods for all 23 datasets together with the predictive performance of RUG (last column). These results can be reproduced by `ruxg_cv_others.py` and `ruxg_cv_RUG.py`. Table EC.4 contains the predictive performance, along with other measures for interpretability, for RUX and RUG. These can be obtained by running `ruxg_cv_rux.py` and `ruxg_cv_rug.py`. 

 2. The script `fairruxg_testing.py` is used to test the FairRUX and FairRUG algorithms for the eight datasets `COMPAS`, `adult`, `default`, `law`, `attrition`, `recruitment`, `student`, and `nursery`. To reproduce the results of the upper half of Table EC.5 in our manuscript, the tests are run with `randomState=21`, `maxDepth=3`, `numEstimators=100`, `fairness_metric='dmc'`, and we set the value of `fairness_eps` to 0 for `COMPAS` and default; 0.01 for `attrition`, `recruitment`, and `student`; 0.025 for `adult`, `law`, and `nursery`. Running the file should immediately start the 10-fold cross validation and append the results to .txt files in the folder FairnessResults. To produce the results of the lower half of Table EC.5, we set `fairness_metric='odm'` and `fairness_eps` to 0 for `COMPAS` and `law`; 0.01 for `nursery` and `attritio`; 0.025 for `default`, `recruitment`, and `student`; 0.05 for `adult`. This file is also used to produce the results of Table 3 in our manuscript, for the `COMPAS` case study.

 3. **TO BE COMPLETED ...** The folder `LoanCase` includes...

 4. The folder `COMPASCase` contains a notebook and two .csv files of results subject to different fairness metrics, used to create Figure 5 in our manuscript. The notebook is ready to run. For Figures 5(a) and 5(b) subject to disparate mistreatment per class, we use `compas_results_dmc.csv`. For Figures 5(c) and 5(d) subject overall disparate mistreatment, we use `compas_results_odm.csv.`

---
**OPTIONAL:**

To use the Gurobi solver, you need to first install
it. The solver is freely available for academic use. Check the
[related
page](https://www.gurobi.com/academia/academic-program-and-licenses/)
on Gurobi's website.
