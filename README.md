# Data-Efficient Learning with Neural Programs
This repository is the official implementation of `Data-efficient learning with Neural Programs`.

## Requirements

Run the following:

1. Install the dependencies inside a new virtual environment: `bash setup.sh`

2. Activate the virtual environment: `conda activate ISED`

3. (Optional) Install package for Neural-GPT experiments: `pip install openai`

### Datasets
* Leaf Identification: download the [leaf dataset](TBD) and place it under `data/leaf_11`.

* Scene Recognization: download the [scene dataset](TBD) and place it under `data/scene`.

* Hand-written Formula: download the [hwf dataset](https://drive.google.com/file/d/1G07kw-wK-rqbg_85tuB7FNfA49q8lvoy/view) and place it under `data/hwf`.

* Visual Sudoku: download the [SatNet dataset](https://powei.tw/sudoku.zip), unzip the data, and place `features.pt`, `features_img.pt`, `labels.pt`, and `perm.pt` under `data/original_data`. 

## Experiments
To reproduce custom ISED experiments in the paper, run 
```bash
cd custom/<TASK>
python PATH_TO_PROGRAM.py
```

To reproduce ISED MNIST-R experiments, run
```bash
cd generation-pipeline
python run.py --task <TASK>
```
where possible task names are sum_2_mnist, sum_3_mnist, sum_4_mnist, less_than_mnist, eq_2_mnist, mod_2_mnist, add_mod_3_mnist, add_sub_mnist, mult_2_mnist, not_3_or_4_mnist, how_many_3_or_4_mnist.

To reproduce experiements for the baselines, we provide additional instructions for [A-NeSI](baselines/a-nesi/readme.md), [DeepProbLog](baselines/dpl/readme.md), [NASR](baselines/nasr/readme.md), [REINFORCE, IndeCateR](baselines/reinforce/readme.md), and [Scallop](baselines/scallop/readme.md).


## Acknowledgements
* Leaf dataset comes from [A Data Repository of Leaf Images: Practice towards Plant Conversarion with Plant Pathology](https://ieeexplore.ieee.org/document/9036158) 
* Scene dataset comes from [Multi-Illumination Dataset](https://projects.csail.mit.edu/illumination/databrowser/index-by-type.html#)