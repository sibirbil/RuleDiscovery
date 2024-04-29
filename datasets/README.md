# Datasets

This directory stores all the datasets used in the manuscript. 

## Binary versions of the datasets

While our method handles continuous and binary features, other methods that we compared to do not and hence we needed to create a binary version of each file. The code to do so can be found in `data_processing.py` in this directory. 

Running this file creates a binary version of each dataset and creates splits used in the cross-validation process. 

After running this file, this folder should have the following structure: 

     .
    ├── datasets/     # datasets used for tutorials and assignments will be made available here
        ├── original/
        ├── binary/
        └── train-test-splits/
            ├── original/
            └── binary/
