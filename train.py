import os
import pickle
import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
import utils
import utils_train
from resnet import ResNet
from inception import Inception
from inception import InceptionTime
from convnet import ConvNet

EPOCHS = 15
LR = 1e-3


def launch_case(**kwargs):
    path_res: str = kwargs["path_res"]

    assert "x_file_name" in kwargs.keys(), "x_file_name has to be specified"
    x_file_name: str = kwargs["x_file_name"]

    assert "path_data" in kwargs.keys(), "path_data has to be specified"
    path_x: str = kwargs["path_data"] + x_file_name

    # assert "y_file_name" in kwargs.keys(), "y_file_name has to be specified"
    # y_file_name: str = kwargs["y_file_name"]

    # assert "path_labels" in kwargs.keys(), "path_labels has to be specified"
    # path_y: str = kwargs["path_labels"] + y_file_name

    for case in kwargs["cases"]:
        print(f"Started training for case: {case}\n")
        try:
            dir_res = path_res + case + os.sep
            os.makedirs(dir_res, exist_ok=True)
        except Exception as e:
            print(f"Failed to create dir: {e}")
            dir_res = Path(os.getcwd()).resolve()

        classifier: str = kwargs["classifier"]
        model: dict = kwargs["classifiers"][classifier]

        # Xy = utils.load_transpose_CER(path_x_train, path_y_train)
        # Xy = Xy.sample(frac=FRAC, random_state=0)
        # X, y = utils.create_X_y_out_of_df(Xy)
        # X = X[:, :17520]
        # y = y[:17520, :]

        assert "path_labels" in kwargs.keys(), "path_labels has to be specified"
        path_y: str = kwargs["path_labels"] + case + ".csv"

        X = utils.create_X_out_of_df(pd.read_csv(path_x))
        # X = X[:, :17520]
        y = utils.create_y_out_of_df(pd.read_csv(path_y))
        # y = y[:17520, :]

        for seed in range(1):
            np.random.seed(seed)
            checkpoint_path = dir_res + classifier + "_" + str(seed) + os.sep + classifier + "_" + str(seed)
            fig_path = dir_res + classifier + "_" + str(seed) + os.sep
            metrics_path = dir_res + classifier + "_" + str(seed) + os.sep
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=seed, shuffle=True
            )
            X_train, X_eval, y_train, y_eval = train_test_split(
                X_train, y_train, test_size=0.1, random_state=seed, shuffle=True
            )
            print(
                f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_eval: {X_eval.shape}, y_eval: {y_eval.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}\n"
            )
            if (
                classifier == "ResNet"
                or classifier == "ConvNet"
                or classifier == "ResNetAtt"
            ):
                launch_deep_training(
                    model=model,
                    xtrain=X_train,
                    xeval=X_eval,
                    xtest=X_test,
                    ytrain=y_train,
                    yeval=y_eval,
                    ytest=y_test,
                    checkpoint_path=checkpoint_path,
                    fig_path=fig_path,
                    metrics_path=metrics_path,
                    classifier_name=classifier,
                    case_name=case,
                )
            elif classifier == "Inception":
                path_inception = utils.create_dir(checkpoint_path)
                # ==================== Ensemble of Inception training ===================#
                for i in range(1):
                    if not utils.check_file_exist(
                        path_inception + "Inception" + str(i) + ".pt"
                    ):
                        launch_deep_training(
                            model=model,
                            xtrain=X_train,
                            xeval=X_eval,
                            xtest=X_test,
                            ytrain=y_train,
                            yeval=y_eval,
                            ytest=y_test,
                            path_to_save=path_inception + "Inception" + str(i),
                        )

                launch_sktime_training(
                    InceptionTime(Inception(), path_inception, 5),
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    checkpoint_path,
                )
            else:
                raise NotImplementedError


def launch_deep_training(**kwargs):
    X_train: np.ndarray = kwargs["xtrain"]
    y_train: np.ndarray = kwargs["ytrain"]
    X_train, y_train = utils.RandomUnderSampler_(X_train, y_train)
    if X_train is None or y_train is None:
        return
    print(f"RandomUnderSampler applied for X_train and y_train\n")
    X_eval: np.ndarray = kwargs["xeval"]
    y_eval: np.ndarray = kwargs["yeval"]
    X_test: np.ndarray = kwargs["xtest"]
    y_test: np.ndarray = kwargs["ytest"]

    model: dict = kwargs["model"]
    model_instance: ResNet = model["model_inst"]
    train_dataset = utils_train.TSDataset(X_train, y_train)
    assert "samples" in train_dataset.__dict__ and "labels" in train_dataset.__dict__
    eval_dataset = utils_train.TSDataset(X_eval, y_eval)
    assert "samples" in eval_dataset.__dict__ and "labels" in eval_dataset.__dict__
    test_dataset = utils_train.TSDataset(X_test, y_test)
    assert "samples" in test_dataset.__dict__ and "labels" in test_dataset.__dict__

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model["batch_size"], shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=model["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_trainer = utils_train.classif_trainer_deep(
        model=model_instance(),
        train_loader=train_loader,
        valid_loader=eval_loader,
        learning_rate=model["lr"],
        weight_decay=model["wd"],
        patience_es=10,
        patience_rlr=5,
        device="mps",
        all_gpu=False,
        verbose=True,
        plotloss=True,
        save_checkpoint=True,
        path_checkpoint=kwargs["checkpoint_path"],
        path_fig=kwargs["fig_path"],
        classifier_name=kwargs["classifier_name"],
        case_name=kwargs["case_name"],
    )

    model_trainer.train(n_epochs=EPOCHS)
    model_trainer.restore_best_weights()
    mean_loss_eval, metrics, y, y_hat = model_trainer.evaluate(
        test_loader, return_output=True
    )
    os.makedirs(kwargs["metrics_path"], exist_ok=True)
    with open(kwargs["metrics_path"] + "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open(kwargs["metrics_path"] + "metrics.txt", "w") as f:
        pprint(metrics)
        print("\n")
        for k, v in metrics.items():
            f.write("{}: {}\n".format(k, v))


def launch_sktime_training(model, X_train, y_train, X_test, y_test, path_res):
    if not utils.check_file_exist(path_res + ".pt"):
        # Equalize class for training
        X_train, y_train = utils.RandomUnderSampler_(X_train, y_train)

        sk_trainer = utils_train.classif_trainer_sktime(
            model.reset(),
            verbose=False,
            save_model=False,
            save_checkpoint=True,
            path_checkpoint=path_res,
        )

        sk_trainer.train(X_train, y_train)
        sk_trainer.evaluate(X_test, y_test)

    return


if __name__ == "__main__":
    root = Path(os.getcwd()).resolve()

    # file_name_train = "xT_residential_25728.csv"
    # path_data_train = str(root) + "/data/CER/data/"
    # path_labels_train = str(root) + "/data/CER/labels/"

    path_data = str(root) + "/data/"
    x_file_name = "pivoted_15.csv"

    path_labels = str(root) + "/labels/"
    # y_file_name = "boiler.csv"

    path_res = str(root) + "/result/"

    case = "boiler"
    cases = [case] if len(sys.argv) <= 2 else sys.argv[2].split(",")
    print(f"\nCases: {cases}\n")

    classifier = "ResNet" if len(sys.argv) <= 1 else str(sys.argv[1])
    print(f"Classifier: {classifier}\n")

    assert classifier, "Classifier has to be specified"
    classifiers = {
        "ResNet": {"model_inst": ResNet, "batch_size": 32, "lr": LR, "wd": 0},
        "Inception": {"model_inst": Inception, "batch_size": 32, "lr": LR, "wd": 0},
        "ConvNet": {"model_inst": ConvNet, "batch_size": 32, "lr": LR, "wd": 0},
    }

    launch_case(
        classifier=classifier,
        classifiers=classifiers,
        cases=cases,
        path_res=path_res,
        # path_data_train=path_data_train,
        # path_labels_train=path_labels_train,
        # file_name_train=file_name_train,
        x_file_name=x_file_name,
        # file_name_test_2=file_name_test_2,
        path_data=path_data,
        # path_data_test_2=path_data_test_2,
        path_labels=path_labels,
        # path_labels_test_2=path_labels_test_2,
        # y_file_name=y_file_name,
    )
