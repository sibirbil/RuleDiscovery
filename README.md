# Rule Generation for Classification: Scalability, Interpretability, and Fairness

**Adia Lumadjeng, Tabea Röber, M. Hakan Akyüz, and Ş. Ilker Birbil**


We introduce a new rule-based optimization method for classification with constraints. The proposed method
takes advantage of linear programming and column generation, and hence, is scalable to large datasets. Moreover, the
method returns a set of rules along with their optimal weights indicating the importance of each rule for learning. Through
assigning cost coefficients to the rules and introducing additional constraints, we show that one can also consider inter-
pretability and fairness of the results. We test the performance of the proposed method on a collection of datasets and
present two case studies to elaborate its different aspects. Our results show that a good compromise between interpretabil-
ity and fairness on the one side, and accuracy on the other side, can be obtained by the proposed rule-based learning
method.

You can find the details of both algorithms in [our manuscript](https://arxiv.org/abs/2104.10751).

This [notebook](RuleDiscovery.ipynb) illustrates how to use RUX and RUG.

## Installation

 1. Install [Anaconda Distribution](https://www.anaconda.com/products/individual).

 2. Create a new environment and install the necessary packages:

 `conda create -n rulediscovery --channel=conda-forge python=3.8 numpy pandas scikit-learn cvxpy cvxopt gurobi`

 3. Activate the current environment:

 `conda activate rulediscovery`

 4. Check whether the installation works:

 `python RUX_RUG_tests.py`

---

**OPTIONAL:**

To use the Gurobi solver, you need to first install
it. The solver is freely available for academic use. Check the
[related
page](https://www.gurobi.com/academia/academic-program-and-licenses/)
on Gurobi's website.
