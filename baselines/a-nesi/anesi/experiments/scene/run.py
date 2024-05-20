import argparse
import time

import yaml
import torch
import wandb

from inference_models import NoPossibleActionsException
from experiments.scene.anesi_scene_id import SceneModel
from experiments.scene.dataset import scene_loader, prepare_inputs

SWEEP = True
file_dict = {}

def test(x, target, box_len, cls, conf, model):
    try:
        test_result = model.test(x, target, box_len, cls, conf)
    except NoPossibleActionsException:
        print("No possible actions during testing")
    acc = test_result[0].item()
    acc_prior = test_result[1].item()
    return acc, acc_prior

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "DEBUG": False,
        "N" : 1,
        "y_encoding": "base6",
        "w_encoding": "base6",
        "model": "full",
        "test": False,
        "batch_size": 16,
        "batch_size_test": 16,
        "amount_samples": 100,
        "predict_only": True,
        "use_prior": True,
        "q_lr": 1e-4,
        "q_loss": "bce",
        "policy": "off",
        "perception_lr": 1e-4,
        "perception_loss": "log-q",
        "percept_loss_pref": 1.,
        "epochs": 100,
        "log_per_epoch": 5,
        "layers": 1,
        "hidden_size": 200,
        "prune": False,
        "dirichlet_init": 1,
        "dirichlet_lr": 0.01,
        "dirichlet_iters": 10,
        "dirichlet_L2": 100000.0,
        "K_beliefs": 100,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config

    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="scene", entity="blackbox-learning")
        config = wandb.config
        print(config)
    elif SWEEP:
        with open("anesi/experiments/leaves/sweeps/sweep.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config.update(wandb.config)
        print(config)
    
    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = SceneModel(config).to(device)

    train_loader, val_loader = scene_loader(config["batch_size"])

    log_iterations = len(train_loader) // config["log_per_epoch"]

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0
        train_acc = 0
        train_acc_prior = 0

        start_epoch_time = time.time()

        for i, (input, file, target) in enumerate(train_loader):
            box_len, cls, data, conf = prepare_inputs(input, file, file_dict)
            target = target.to(device)
            try:
                trainresult = model.train_all(data, target, box_len, cls, conf)
                loss_percept = trainresult.percept_loss
                loss_nrm = trainresult.q_loss
            except NoPossibleActionsException:
                print("No possible actions during training")
                continue
            
            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            test_result = test(data, target, box_len, cls, conf, model)
            train_acc += test_result[0]
            train_acc_prior += test_result[1]

            if (i + 1) % log_iterations == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean()

                log_q_weight = torch.sigmoid((config['percept_loss_pref'] * avg_alpha).log()).item()
                avg_alpha = avg_alpha.item()

                print(f"actor: {cum_loss_percept / log_iterations:.4f} "
                      f"nrm: {cum_loss_nrm / log_iterations:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f} ",
                      f"log_q_weight: {log_q_weight:.4f} ",
                      f"train_acc: {train_acc / log_iterations:.4f}",
                      f"train_acc_prior: {train_acc_prior / log_iterations:.4f}")

                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "nrm_loss": cum_loss_nrm / log_iterations,
                    "train_accuracy": train_acc / log_iterations,
                    "train_accuracy_prior": train_acc_prior / log_iterations,
                    "avg_alpha": avg_alpha,
                    # "log_q_weight": log_q_weight,
                })
                cum_loss_percept = 0
                cum_loss_nrm = 0
                train_acc = 0
                train_acc_prior = 0

            end_epoch_time = time.time()  
            
        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc = 0.
        val_acc_prior = 0.             
        
        for i, (input, file, target) in enumerate(val_loader):
            box_len, cls, input, conf = prepare_inputs(input, file, file_dict)
            target = target.to(device)
            test_result = test(input.to(device), target.to(device), box_len, cls, conf, model)
            val_acc += test_result[0]
            val_acc_prior += test_result[1]

        val_accuracy = val_acc / len(val_loader)
        val_accuracy_prior = val_acc_prior / len(val_loader)
        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy}",
              f"Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })

