import grid_search_helpers as gs_helpers
import Datasets as DS

binary = True
write = True
numCV: int = 5
testSize = 0.2
randomState = 21

problems = [DS.banknote, DS.hearts, DS.ILPD, DS.ionosphere,
            DS.liver, DS.diabetes_pima, DS.tictactoe, DS.transfusion,
            DS.wdbc, DS.adult, DS.bank_mkt, DS.magic, DS.mushroom, DS.musk,
            DS.oilspill, DS.phoneme, DS.mammography, DS.skinnonskin]

CG_pgrid = {'epsilon':[1]}

complexities = {
    'banknote':[20,25,30],
    'hearts':[10,15,20],
    'ILPD':[5,10,15],
    'ionosphere':[10,15,20],
    'liver':[5,10,15],
    'diabetes_pima':[3,5,10],
    'tictactoe':[30,35,40],
    'transfusion':[5,10,15],
    'wdbc':[10,15,20],
    'adult':[85,90,95],
    'bank_mkt':[5,10,15],
    'magic':[90,95,100],
    'mushroom':[15,20,25],
    'musk':[120,125,130],
    'oilspill':[10,20,30],
    'phoneme':[10,20,30],
    'mammography':[10,20,30],
    'skinnonskin':[10,20,30]
}

for problem in problems:
    pname = problem.__name__
    CG_pgrid['complexity'] = complexities[pname]
    print(CG_pgrid)

    gs_helpers.run(problem, CG_pgrid, model = 'CG',
                randomState = randomState, testSize=testSize, numSplits=numCV, binary = binary, write=write,
                   data_path='./prepped_data_CG/')