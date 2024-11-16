import logging

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

import json
import argparse

from transformers import AutoTokenizer
from paraphrase.utils.labelexpand import *
from model import ContrastNet

from paraphrase.utils.data import FewShotDataset, FewShotSSLFileDataset

from utils.data import get_json_data, FewShotDataLoader
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict, Callable, Union

from tensorboardX import SummaryWriter
import numpy as np
import warnings
from utils.few_shot import create_episode, create_ARSC_test_episode, create_ARSC_train_episode

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_fsintent(
        # Compulsory!
        #数据集名称
        dataset:str,
        #全部数据地址
        data_path: str,
        #标签地址
        train_labels_path: str,
        # Few-shot Stuff
        #支撑集数量 K5
        n_support: int,
        #查询集数量 K5
        n_query: int,
        #类别数量C5
        n_classes: int,
        #encoder预训练模型
        model_name_or_path: str,
        #监督tau
        super_tau: float = 1.0,
        #学习率
        lr: float = 1e-6,
        #计算距离度量
        metric: str = "euclidean",
        logger: object = None,
        #监督损失权重
        super_weight: float = 0.7,
        #单样本最大长度
        max_len: int = 64,
        # Path training data ONLY (optional)
        #训练集地址 train_aug
        train_path: str = None,
        # Validation & test
        valid_labels_path: str = None,
        test_labels_path: str = None,
        #训练100次后在valid上评估
        evaluate_every: int = 100,
        #训练1000次后在test上评估
        n_test_episodes: int = 1000,

        # Logging & Saving
        #输出地址
        output_path: str = f'runs/{now()}',
        model_best_path: str=None,
        #训练10次后输出
        log_every: int = 10,

        # Training stuff
        #最大迭代次数
        max_iter: int = 10000,
        early_stop: int = None,





):
    if output_path:
        if os.path.exists(output_path) and len(os.listdir(output_path)):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    # ----------
    # Load model
    # ----------

    fsinet: ContrastNet = ContrastNet(config_name_or_path=model_name_or_path, metric=metric, max_len=max_len, super_tau=super_tau,
                                      n_classes=n_classes,n_support=n_support,iterations=3)
    optimizer = torch.optim.Adam(fsinet.parameters(), lr=lr)

    #logger.info(torch.cuda.list_gpu_processes())

    # ------------------
    # Load Train Dataset
    # ------------------

    #获取标签扩展信息

    labeldict = Labelexpand(dataset).getlabeldict()
    train_dataset = FewShotSSLFileDataset(
        data_path=train_path if train_path else data_path,
        labels_path=train_labels_path,
        n_classes=n_classes,
        n_support=n_support,
        n_query=n_query,
        labeldict=labeldict
    )

    logger.info(f"Train dataset has {len(train_dataset)} items")

    # ---------
    # Load data
    # ---------
    logger.info(f"train labels: {train_dataset.data.keys()}")
    valid_dataset: FewShotDataset = None
    if valid_labels_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
        valid_dataset = FewShotDataset(data_path=data_path, labels_path=valid_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query,labeldict=labeldict)
        logger.info(f"valid labels: {valid_dataset.data.keys()}")
        assert len(set(valid_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    test_dataset: FewShotDataset = None
    if test_labels_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()
        test_dataset = FewShotDataset(data_path=data_path, labels_path=test_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query,labeldict=labeldict)
        logger.info(f"test labels: {test_dataset.data.keys()}")
        assert len(set(test_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0
    best_valid_dict = None
    best_test_dict = None


    #开始训练
    for step in range(max_iter):
        #获取元任务数据
        episode = train_dataset.get_episode()
        supervised_loss_share = super_weight
        loss, loss_dict = fsinet.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share)
        fsinet.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share)


        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            for key, value in train_metrics.items():
                train_writer.add_scalar(tag=key, scalar_value=np.mean(value), global_step=step)
            logger.info("epsisode "+str(step+1)+" | "+f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": key,
                        "value": np.mean(value)
                    }
                    for key, value in train_metrics.items()
                ],
                "global_step": step
            })

            train_metrics = collections.defaultdict(list)

        if valid_labels_path or test_labels_path:
            if (step + 1) % evaluate_every == 0:
                is_best = False
                for labels_path, writer, set_type, set_dataset in zip(
                        [valid_labels_path, test_labels_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_dataset, test_dataset]
                ):
                    if set_dataset:
                        #test过程
                        set_results = fsinet.test_step(
                            dataset=set_dataset,
                            n_episodes=n_test_episodes
                        )


                        for key, val in set_results.items():
                            writer.add_scalar(tag=key, scalar_value=val, global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": key,
                                    "value": val
                                }
                                for key, val in set_results.items()
                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in set_results.items()]))
                        if set_type == "valid":
                            if set_results["acc_induction"] >= best_valid_acc:
                                best_valid_acc = set_results["acc_induction"]
                            # if set_results["acc"] >= best_valid_acc:
                            #     best_valid_acc = set_results["acc"]
                                best_valid_dict = set_results
                                is_best = True
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")
                        else:
                            if is_best:
                                best_test_dict = set_results
                                torch.save(fsinet,model_best_path)

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    logger.info(f"Best eval results: ")
                    logger.info(f"valid | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_valid_dict.items()]))
                    logger.info(f"test | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_test_dict.items()]))
                    break

    logger.info(f"Best eval results: ")
    logger.info(f"valid | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_valid_dict.items()]))
    logger.info(f"test | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_test_dict.items()]))

    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)

    fsinet=torch.load(model_best_path)
    fsinet.test_step(dataset=test_dataset,n_episodes=n_test_episodes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the full data")
    parser.add_argument("--train-labels-path", type=str, required=True, help="Path to train labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--train-path", type=str, help="Path to training data (if provided, picks training data from this path instead of --data-path")
    parser.add_argument("--model-name-or-path", type=str, default='bert-base-uncased', help="Language Model PROTAUGMENT initializes from")
    parser.add_argument("--lr", default=1e-6, type=float, help="Temperature of the contrastive loss")
    parser.add_argument("--super-tau", default=1.0, type=float, help="Temperature of the contrastive loss in supervised loss")


    parser.add_argument("--super-weight", default=0.7, type=float, help="The initialized supervised loss weight")

    parser.add_argument("--max-len", type=int, default=64, help="maxmium length of text sequence for BERT") 

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=1, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=1, help="Number of classes per episode")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance function to use", choices=("euclidean", "cosine"))
    parser.add_argument("--n-task", type=int, default=5, help="Number of tasks in task-level regularizer")

    # Validation & test
    parser.add_argument("--valid-labels-path", type=str, required=True, help="Path to valid labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--test-labels-path", type=str, required=True, help="Path to test labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--n-test-episodes", type=int, default=600, help="Number of episodes during evaluation (valid, test)")

    # Logging & Saving
    parser.add_argument("--output-path", type=str, default=f'runs/{now().replace(":","-")}')
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--log-path", type=str, help="Path to the log file.")
    parser.add_argument("--model-best-path", type=str, help="Path to save the best model.")

    # Training stuff
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--early-stop", type=int, default=10, help="Number of worse evaluation steps before stopping. 0=disabled")







    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    warnings.simplefilter('ignore')

    handler = logging.FileHandler(args.log_path, mode='w')

    logger.addHandler(handler)

    logger.debug(f"Received args: {json.dumps(args.__dict__, sort_keys=True, ensure_ascii=False, indent=1)}")

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.data_path, args.train_labels_path, args.valid_labels_path, args.test_labels_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_fsintent(
        dataset=args.dataset,
        data_path=args.data_path,
        train_labels_path=args.train_labels_path,
        train_path=args.train_path,
        model_name_or_path=args.model_name_or_path,
        super_tau=args.super_tau,
        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        metric=args.metric,
        logger=logger,
        super_weight=args.super_weight,
        max_len=args.max_len,

        valid_labels_path=args.valid_labels_path,
        test_labels_path=args.test_labels_path,
        evaluate_every=args.evaluate_every,
        n_test_episodes=args.n_test_episodes,

        output_path=args.output_path,
        log_every=args.log_every,
        model_best_path=args.model_best_path,
        max_iter=args.max_iter,
        early_stop=args.early_stop,
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()


