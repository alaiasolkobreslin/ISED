# REINFORCE / IndeCateR

## Experiments
To reproduce the experiments for REINFORCE and IndeCateR, set the variable `grad_type` to either `reinforce` or `icr` and run
```bash
cd baselines/reinforce
python <TASK>.py
```

## Paper
*Differentiable Sampling of Categorical Distributions Using the CatLog-Derivative Trick [[paper](https://arxiv.org/abs/2311.12569)] [[github](https://github.com/ML-KULeuven/catlog)]