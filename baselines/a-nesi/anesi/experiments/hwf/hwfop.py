import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch
import wandb

from inference_models import NoPossibleActionsException
from experiments.hwf.anesi_hwf_run import HWFModel
from experiments.hwf.dataset import hwf_loader

SWEEP = True

def test(x, label, eqn_len, model):
    try:
        test_result = model.test(x, label, eqn_len)
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
        "y_encoding": "base10",
        "w_encoding": "base10",
        "model": "full",
        "test": False,
        "batch_size": 16,
        "batch_size_test": 16,
        "amount_samples": 100,
        "predict_only": True,
        "use_prior": True,
        "q_lr": 1e-4,
        "q_loss": "mse",
        "policy": "off",
        "perception_lr": 1e-4,
        "perception_loss": "log-q",
        "percept_loss_pref": 1.,
        "epochs": 100,
        "log_per_epoch": 10,
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

        run = wandb.init(config=config, project="hwf", entity="seewonchoi")
        config = wandb.config
        print(config)
    elif SWEEP:
        # TODO: I don't get how it's supposed to know what yaml file to open here.
        with open("anesi/experiments/hwf/sweeps/sweep.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config.update(wandb.config)
        print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = HWFModel(config).to(device)

    train_loader, val_loader = hwf_loader(config["batch_size"], config["batch_size_test"])

    print(len(val_loader))

    log_iterations = len(train_loader) // config["log_per_epoch"]

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0
        train_acc = 0
        train_acc_prior = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            # label_digits is ONLY EVER to be used during testing!!!
            data, eqn_len, target = batch
            x = data.to(device)
            label = target.to(device)

            try:
                trainresult = model.train_all(x, eqn_len, label)
                loss_percept = trainresult.percept_loss
                loss_nrm = trainresult.q_loss
            except NoPossibleActionsException:
                print("No possible actions during training")
                continue

            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            test_result = test(x, label, eqn_len, model)
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
        for i, batch in enumerate(val_loader):
            data, eqn_len, target = batch

            test_result = test(data.to(device), label.to(device), eqn_len, model)
            val_acc += test_result[0]
            val_acc_prior += test_result[1]

        val_accuracy = val_acc / len(val_loader)
        val_accuracy_prior = val_acc_prior / len(val_loader)
        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy} {prefix}",
              f"Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })





