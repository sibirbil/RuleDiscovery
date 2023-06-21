import Datasets as DS
import grid_search_helpers as gs_helpers

binary = False
write = True
numCV: int = 5
testSize = 0.2
randomState = 21
fairness_metric = 'odm'
fairness_metric = 'dmc'

# CLASSIFICATION

problems = [DS.compas, DS.default, DS.law, DS.attrition, DS.recruitment, DS.student, DS.nursery]


RUG_pgrid = {'pen_par': [0.1, 1.0, 10.0],
             'max_depth': [3, 5],
             'max_RMP_calls': [5, 10, 15],
             'fair_eps':[0.01, 0.025, 0.05, 0.1]}

####################
# Solve all problems

for problem in problems:
    gs_helpers.run(problem, RUG_pgrid, model = 'FairRUG',
                   randomState = randomState, testSize=testSize, numSplits=numCV,
                   binary = binary, write=write, fairness_metric = fairness_metric)
