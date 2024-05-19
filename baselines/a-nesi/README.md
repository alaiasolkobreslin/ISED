# A-NeSI: Approximate Neurosymbolic Inference

Run the following:

1. `cd baselines/a-nesi`

2. Install additional dependencies inside the virtual environment: `bash setup_anesi.sh`

3. Install the A-NeSI module:  `pip install -e .`

## Experiments
The experiments are organized with Weights&Biases. To reproduce the experiments from the paper, run
```bash
wandb sweep anesi/experiments/<TASK>/repeat/test_predict_only.yaml
wandb agent <sweep_id>
```
Note that you will need to update the entity and project parameters of wandb in the sweep files. 

## Paper
Arxiv: [A-NeSI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference](https://arxiv.org/abs/2212.12393)

GitHub: https://github.com/HEmile/a-nesi