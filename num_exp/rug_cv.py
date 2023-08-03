import Datasets as DS
import grid_search_helpers as gs_helpers

binary = False
write = True
numCV: int = 5
testSize = 0.2
randomState = 21
RUG_rule_length_cost = True
RUG_threshold = 0.05

# for fairness results
# RUG_rule_length_cost = False
# RUG_threshold = None
# RUG_record_fairness = True

# CLASSIFICATION
# binary
problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

# multiclass
# problems = [DS.wine, DS.glass, DS.ecoli, DS.sensorless, DS.seeds]

# fairness results
# problems = [DS.adult, DS.compas, DS.default, DS.attrition, DS.nursery, DS.student, DS.law]

RUG_pgrid = {'pen_par': [0.1, 1.0, 10.0],
             'max_depth': [3, 5],
             'max_RMP_calls': [5, 10, 15]}

####################
# Solve all problems

for problem in problems:
    gs_helpers.run(problem, RUG_pgrid, model = 'RUG',
                   randomState = randomState, testSize=testSize, numSplits=numCV,
                   binary = binary, write=write, RUG_rule_length_cost=RUG_rule_length_cost,
                   RUG_threshold=RUG_threshold, datasets_path='./datasets/datasets/', save_path='./results_w_RUG/')
