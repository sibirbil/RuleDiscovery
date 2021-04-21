# Discovering Classification Rules for Interpretable Learning with Linear Programming

**Hakan Akyüz & İlker Birbil**

Rules embody a set of if-then statements which include one or more conditions to classify a subset of samples in a dataset. In various applications such classification rules are considered to be interpretable by the decision makers. We introduce two new algorithms for interpretability and learning. Both algorithms take advantage of linear programming, and hence, they are scalable to large data sets. The first algorithm (RUX) extracts rules for interpretation of trained models that are based on tree/rule ensembles. The second algorithm (RUG) generates a set of classification rules through a column generation approach. The proposed algorithms return a set of rules along with their optimal weights indicating the importance of each rule for classification.  Moreover, our algorithms allow assigning cost coefficients, which could relate to different attributes of the rules, such as; rule lengths, estimator weights, number of false negatives, and so on.  Thus, the decision makers can adjust these coefficients to divert the training process and obtain a set of rules that are more appealing for their needs.

## Installation

 1. Install Anaconda Distribution (https://www.anaconda.com/products/individual)

 2. Create a new environment and install the necessary packages:

 `conda create -n rulediscovery --channel=conda-forge python=3.8 numpy pandas scikit-learn cvxpy cvxopt gurobi`

 3. Activate the current environment:

 `conda acivate rulediscovery`

 4. Check whether the installation works:

 `python RUX_RUG_tests.py`

_OPTIONAL:_ To be able to use the Gurobi solver, you need to first
install it. The solver is freely available for academic use. See
[Gurobi
page.](https://www.gurobi.com/academia/academic-program-and-licenses/)
