import Datasets as DS
import grid_search_helpers as gs_helpers

binary = False
write = True
numCV: int = 5
testSize = 0.2
randomState = 21

# CLASSIFICATION
# binary
problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

# multiclass
# problems = [DS.wine, DS.glass, DS.ecoli, DS.sensorless, DS.seeds]

RUG_pgrid = {'pen_par': [0.1, 1.0, 10.0],
             'max_depth': [3, 5],
             'max_RMP_calls': [5, 10, 15]}

####################
# Solve all problems

for problem in problems:
    gs_helpers.run(problem, RUG_pgrid, model = 'RUG',
                   randomState = randomState, testSize=testSize, numSplits=numCV,
                   binary = binary, write=write)