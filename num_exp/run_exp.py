import Datasets as DS
import sys
# sys.path.insert(1, './num_exp/helpers')
import grid_search_helpers as gs_helpers

'''
Below please specify the model you want to run. Further down you may need to adjust some parameters. 
For example, you could choose other datasets or an alternative fairness metrics used for FairRUG and FairCG. 
Hence, please check the options for each model in their section down below. 
'''

# -----------
# specify which model to run
# -----------
# model = 'RUG'
model = 'DT-heuristic'
# any of 'RUG', 'FairRUG', 'FSDT', 'CG', 'FairCG', 'binoct'

# -----------
# specify parameters for reproducibility
# -----------
write = True
numCV: int = 5
testSize = 0.2
randomState = 21


# -----------
# RUG
# -----------
if model == 'RUG':
    # for comparison to RUG exact
    # binary = True
    # RUG_rule_length_cost = False

    # for interpretability results (Table 9)
    binary = False
    RUG_rule_length_cost = True
    RUG_threshold = 0.05
    RUG_record_fairness = False

    # fairness results (Table 5)
    # binary = False
    # RUG_rule_length_cost = False
    # RUG_threshold = None
    # RUG_record_fairness = True

    # datasets
    # binary
    problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
                DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
                DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
                DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]
    problems = [DS.banknote]

    # multiclass (Table 2)
    # problems = [DS.wine, DS.glass, DS.ecoli, DS.sensorless, DS.seeds]

    # fairness (Table 5)
    # problems = [DS.attrition, DS.student, DS.nursery, DS.law]

    # specify parameters for grid search
    RUG_pgrid = {'pen_par': [0.1, 1.0, 10.0],
                 'max_depth': [3, 5],
                 'max_RMP_calls': [5, 10, 15]}

    # solve problems
    for problem in problems:
        gs_helpers.run(problem, RUG_pgrid, model = 'RUG',
                       randomState = randomState, testSize=testSize, numSplits=numCV,
                       binary = binary, write=write, RUG_rule_length_cost=RUG_rule_length_cost,
                       RUG_threshold=RUG_threshold,
                       save_path='./results/results_w_RUG/',
                       RUG_record_fairness=RUG_record_fairness)


# -----------
# FairRUG
# -----------
elif model == 'FairRUG':
    binary = False

    # for fairness results
    fairness_metric = 'odm'
    # fairness_metric = 'dmc'
    # fairness_metric = 'EqOpp'
    RUG_rule_length_cost = False
    RUG_threshold = None
    RUG_record_fairness = True

    # datasets
    # dmc and EqOpp (for binary classification)
    # problems = [DS.adult, DS.compas, DS.default]

    # dmc and odm (for multiclass and multiple groups)
    problems = [DS.attrition, DS.nursery, DS.student, DS.law]

    # specify parameters for grid search
    RUG_pgrid = {'pen_par': [0.1, 1.0, 10.0],
                 'max_depth': [3, 5],
                 'max_RMP_calls': [5, 10, 15],
                 'fair_eps': [0, 0.01, 0.025, 0.05]}

    for problem in problems:
        gs_helpers.run(problem, RUG_pgrid, model='FairRUG',
                       randomState=randomState, testSize=testSize, numSplits=numCV,
                       binary=binary, write=write, fairness_metric=fairness_metric,
                       save_path='./results/results_w_FairRUG/')

# -----------
# FSDT
# -----------
elif model == 'FSDT':
    binary=True

    # datasets
    # binary
    problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
                DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
                DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
                DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

    # parameters for grid search
    DL85_pgrid = {'max_depth': [3, 5, 10]}

    # solve problems
    for problem in problems:
        gs_helpers.run(problem, DL85_pgrid, model='FSDT', randomState=randomState, testSize=testSize,
                       numSplits=numCV, binary=binary, write=write, save_path='./results/results_w_FSDT/')

# -----------
# CG
# -----------
elif model == 'CG':
    binary = True

    # datasets
    # binary
    problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
                DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
                DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
                DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

    # parameters for grid search
    CG_pgrid = {'epsilon': [1]}

    complexities = {
        'banknote': [20, 25, 30],
        'hearts': [10, 15, 20],
        'ILPD': [5, 10, 15],
        'ionosphere': [10, 15, 20],
        'liver': [5, 10, 15],
        'diabetes_pima': [3, 5, 10],
        'tictactoe': [30, 35, 40],
        'transfusion': [5, 10, 15],
        'wdbc': [10, 15, 20],
        'adult': [85, 90, 95],
        'bank_mkt': [5, 10, 15],
        'magic': [90, 95, 100],
        # 'mushroom': [15, 20, 25],
        'mushroom':[5,10],
        'musk': [120, 125, 130],
        'oilspill': [10, 20, 30],
        'phoneme': [10, 20, 30],
        'mammography': [10, 20, 30],
        'skinnonskin': [10, 20, 30],
        'compas': [5, 10, 15],
        'default': [5, 10, 15]
    }

    # solve problems
    for problem in problems:
        pname = problem.__name__
        CG_pgrid['complexity'] = complexities[pname]

        gs_helpers.run(problem, CG_pgrid, model='CG', randomState=randomState, testSize=testSize,
                       numSplits=numCV, binary=binary, write=write, save_path='./results/results_w_CG/')

# -----------
# FairCG
# -----------
elif model == 'FairCG':
    binary = True

    # adjust fairness metric if desired
    fairness_metric = 'EqOfOp'
    # fairness_metric = 'HammingEqOdd'

    # datasets
    # binary classification datasets with 2 sensitive groups
    problems = [DS.compas, DS.adult, DS.default]

    # parameters for grid search
    CG_pgrid = {'epsilon': [0.025]}

    complexities = {
        'adult': [85, 90, 95],
        'compas': [5, 10, 15],
        'default': [5, 10, 15]
    }

    # solve problems
    for problem in problems:
        pname = problem.__name__
        CG_pgrid['complexity'] = complexities[pname]

        gs_helpers.run(problem, CG_pgrid, model='FairCG', randomState=randomState, testSize=testSize,
                       numSplits=numCV, binary=binary, write=write, fairness_metric=fairness_metric,
                       save_path='./results/results_w_FairCG/')

# -----------
# binoct
# -----------
elif model == 'binoct':
    binary = True

    # datasets
    # binary
    problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
                DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
                DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
                DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

    binoct_pgrid = {'max_depth': [3, 5]}

    # solve problems
    for problem in problems:
        gs_helpers.run(problem, binoct_pgrid, model='binoct', randomState=randomState, testSize=testSize,
                       numSplits=numCV, binary=binary, write=write, save_path='./results/results_w_binoct/')

elif model == 'DT-heuristic':
    binary=False

    # datasets
    # binary
    problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
                DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
                DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
                DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]
    problems = [DS.ionosphere, DS.magic, DS.oilspill, DS.skinnonskin]
    problems = [DS.skinnonskin]
    # problems = [DS.wine, DS.glass, DS.ecoli, DS.sensorless, DS.seeds]
    # problems = [DS.sensorless]

    # parameters for grid search
    DT_heuristic_pgrid = {'max_depth': [3, 5, 10]}
    DT_heuristic_pgrid = {'max_depth': [3,5]}


    # solve problems
    for problem in problems:
        try: gs_helpers.run(problem, DT_heuristic_pgrid, model='DT-heuristic', randomState=randomState, testSize=testSize,
                       numSplits=numCV, binary=binary, write=write, save_path='./results/results_w_DTheuristic/')
        except: continue