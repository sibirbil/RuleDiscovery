import grid_search_helpers as gs_helpers
import Datasets as DS

binary = False
write = True
numCV: int = 5
testSize = 0.2
randomState = 21

problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

binoct_pgrid = {'max_depth':[3,5,10]}

for problem in problems:
    gs_helpers.run(problem, binoct_pgrid, model = 'binoct',
                   randomState = randomState, testSize=testSize, numSplits=numCV,
                   binary = binary, write=write, data_path='./prepped_data_binoct/')

