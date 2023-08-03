import grid_search_helpers as gs_helpers
import Datasets_binary as DS

binary = False
write = True
numCV: int = 5
testSize = 0.2
randomState = 21
fairness_metric = 'EqOfOp' # or 'HammingEqOdd'
# fairness_metric = 'HammingEqOdd'

problems = [DS.compas, DS.adult, DS.default]

CG_pgrid = {'epsilon':[0.025]}

# CG_pgrid = {'complexity':[3, 5, 10]}
complexities = {
    'adult':[85,90,95],
    'compas':[5,10,15],
    'default':[5,10,15]
}

for problem in problems:
    # solve_problem(problem)
    pname = problem.__name__
    CG_pgrid['complexity'] = complexities[pname]
    print(CG_pgrid)

    gs_helpers.run(problem, CG_pgrid, model = 'FairCG',
                randomState = randomState, testSize=testSize, numSplits=numCV, binary = False, write=write,
                   data_path='./prepped_data_CG/', fairness_metric=fairness_metric,
                   datasets_path = './datasets/CG_binarized/', save_path = './results_w_FairCG/')