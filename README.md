# MLSA
This is the implementation of "Multi-Task Learning with Loop Specific Attention for CDR Structure Prediction".

## Local Environment Setup
conda create --n `name` python=3.7 \
conda activate `name`

## Installing project requirements
- CUDA >= 11.1 
- PyTorch == 1.8.2 (LTS Version) 
- Numpy >= 1.18.1 
- tqdm

## Arguments
train_path: Path for training dataset \
val_path: Path for validation dataset \
test_path: Path for test dataset \
hidden_size: hidden layer dimension \
k_neighbors: KNN neighborhood size \
depth: number of message passing layers \
vocab_size: Size of amino acid vocabulary \
dropout: Dropout \
epochs: Epochs \
seed: Random number generator

## Training
To train the model, please run `python main.py` \
During training, the script reports Root Mean Square Deviation (RMSD) on the validation set for each of the three loops.

## Results
After the script completes, RMSD is reported on the test set for each of the three loops.
