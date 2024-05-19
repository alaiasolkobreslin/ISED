# ISED: Infer - Sample - Estimate - Descent
This repository contains the code for the paper `Data-efficient learning with Neural Programs`.

## Requirements

Run the following:

1. Install the dependencies inside a new virtual environment: `bash setup.sh`

2. Activate the virtual environment: `conda activate ISED`

3. (Optional) Install package for Neural-GPT experiments: `pip install openai`

## Experiments
To reproduce the experiments in the paper, run 
```bash
cd custom/<TASK>
python PATH_TO_PROGRAM.py
```

To reproduce experiements for the baselines, we provide additional instructions for
[A-NeSI](baselines/a-nesi/readme.md), [DeepProbLog](baselines/dpl/readme.md), [NASR](baselines/nasr/readme.md), [REINFORCE, Catlog](baselines/reinforce/readme.md), and [Scallop](baselines//readme.md).

### Datasets
* Leaf Identification: download the [leaf dataset](https://drive.google.com/file/d/146WOKq8i9UEXnxD4-pQp9xo_a9kz0UmY/view?usp=share_link) and place it under `data/leaf_11`.

* Scene Recognization: download the scene dataset and place it under `data/scene`.

* Hand-written Formula: download the [hwf dataset](https://drive.google.com/file/d/1klad1oSqzt7gHDibKZnW9mMlB2KBkMNd/view?usp=share_link) and place it under `data/hwf`.

* Visual Sudoku: download the [SatNet dataset](https://powei.tw/sudoku.zip), unzip the data (4 files), and place `features.pt`, `features_img.pt`, `labels.pt`, and `perm.pt` under `data/original_data`. 

## Acknowledgements
Leaf dataset comes from [A Data Repository of Leaf Images: Practice towards Plant Conversarion with Plant Pathology](https://ieeexplore.ieee.org/document/9036158) and scene dataset comes from [Multi-Illumination Dataset](https://projects.csail.mit.edu/illumination/databrowser/index-by-type.html#)