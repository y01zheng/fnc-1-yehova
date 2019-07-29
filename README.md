# Enhanced Baseline FNC implementation
## Getting Started
The FNC dataset is inlcuded as a submodule and can be FNC Dataset is included as a submodule. You should download the fnc-1 dataset by running the following commands. This places the fnc-1 dataset into the folder fnc-1/

    git submodule init
    git submodule update

## run my enhanced baseline
    python3 yehova.py

## run comparative method
    python3 randomForest.py

## tune hyperparameters
Before running, you should set the ranges of hyperparameters according to your concrete requirements. I just provide examples to tune hyperparameters.
    
    python3 auto_tuning_params.py  (using sklearn.model_selection.GridSearchCV)

    or 
    
    sh auto_tuning_scipt.sh (using scipt to run different hyperparameter settings)
