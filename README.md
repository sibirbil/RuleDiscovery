# Rule Generation for Classification: Scalability, Interpretability, and Fairness

**Adia Lumadjeng, Tabea Röber, M. Hakan Akyüz, and Ş. Ilker Birbil**


We introduce a new rule-based optimization method for classification with constraints. The proposed method
takes advantage of linear programming and column generation, and hence, is scalable to large datasets. Moreover, the method returns a set of rules along with their optimal weights indicating the importance of each rule for learning. Through assigning cost coefficients to the rules and introducing additional constraints, we show that one can also consider inter pretability and fairness of the results. We test the performance of the proposed method on a collection of datasets and present two case studies to elaborate its different aspects. Our results show that a good compromise between interpretability and fairness on the one side, and accuracy on the other side, can be obtained by the proposed rule-based learning method.

You can find the details in [our manuscript](https://arxiv.org/abs/2104.10751).

This [notebook](RuleDiscovery.ipynb) illustrates how to use RUX and RUG.

## Installation

 1. Install [Anaconda Distribution](https://www.anaconda.com/products/individual).

 2. Create a new environment and install the necessary packages:

 `conda create -n rulediscovery -c conda-forge numpy pandas scikit-learn cvxpy cvxopt`

 3. Activate the current environment and install `gurobi` package in the environment:

 `conda activate rulediscovery`
 
 `conda install -c gurobi gurobi`
 

## Repo structure

The code contains the following files to reproduce the results of our manuscript:

1. In the jupyter notebook `RuleDiscovery.ipynb` we demonstrate how to use RUG and RUX in single fold on the ecoli dataset. The code will produce the results of RF, ADA and GB, along with RUX(RF), RUX(ADA), RUX(GB), and RUG.
 
2. The folder `num_exp` contains all files for the numerical experiments reported in the manuscript. Please see the `README.md` in that directory for more details.

3. The folder `num_exp/case_study` contains the files for reproducing results of the case study reported in the manuscript. 


## Solvers

**Note that the default for `solver` option is 'gurobi'.** To use the Gurobi solver, you need to first install
it. The solver is freely available for academic use. Check the [related page](https://www.gurobi.com/academia/academic-program-and-licenses/)
on Gurobi's website. The current version of our code also supports the open source solver [GLPK](https://www.gnu.org/software/glpk/) (set `solver='glpk'`).
