mkdir data

conda create --name ised_sudoku python=3.9
conda activate ised_sudoku

mkdir data/original_data

pip install pyswip
conda install joblib
conda install numpy
conda install matplotlib
conda install -c conda-forge mnist
conda install tqdm
conda install h5py

python3.9 -m pip install ipykernel

conda install -c conda-forge optuna