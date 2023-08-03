import Datasets_binary as DS
import grid_search_helpers as gs_helpers

write = True
numCV: int = 5
testSize = 0.2
randomState = 21

# CLASSIFICATION
problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

DL85_pgrid = {'max_depth':[3,5,10]}

####################
# Solve all problems

for problem in problems:
    gs_helpers.run(problem, DL85_pgrid, model = 'FSDT',
                randomState = randomState, testSize=testSize, numSplits=numCV, binary = False, write=write,
                   data_path='./prepped_data_FSDT/', datasets_path = './datasets/FSDT_binarized/',
                   save_path = './results_w_FSDT_binary/')
