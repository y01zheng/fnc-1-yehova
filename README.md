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
    python3 auto_tuning_params.py
