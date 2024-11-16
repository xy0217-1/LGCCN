import numpy as np
import collections
from typing import List, Dict, Callable, Union
import logging
import torch

import random
from torch.utils.data import Dataset
#from transformers.models.auto.tokenization_auto import BartTokenizerFast
from paraphrase.modeling import ParaphraseModel
from utils.data import get_json_data, get_txt_data

#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
# from models.use import USEEmbedder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#fewshot数据处理
class FewShotDataset:
    def __init__(
            self,
            labeldict,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            labels_path: str = None,
    ):
        self.data_path = data_path
        self.labels_path = labels_path
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.data: Dict[str, List[Dict]] = None
        self.counter: Dict[str, int] = None
        self.load_file(data_path, labels_path)
        self.labelsdict=labeldict

    def load_file(self, data_path: str, labels_path: str = None):
        data = get_json_data(data_path)
        if labels_path:
            labels = get_txt_data(labels_path)
        else:
            labels = sorted(set([item["label"] for item in data]))


        #labels_dict将相同label的数据放入同一个字典，比如labels_dict[39]为一个list，存放text
        labels_dict = collections.defaultdict(list)
        for item in data:
            if item["label"] in labels:
                labels_dict[item['label']].append(item)
        labels_dict = dict(labels_dict)

        #将同一label的text打乱
        for key, val in labels_dict.items():
            random.shuffle(val)
        self.data = labels_dict
        self.counter = {key: 0 for key, _ in self.data.items()}

    def get_episode(self) -> Dict:
        episode = dict()
        if self.n_classes:
            assert self.n_classes <= len(self.data.keys())
            #随机从labels中抽取5类label,不抽取相同label
            rand_keys = np.random.choice(list(self.data.keys()), self.n_classes, replace=False)
            # 确保标签数据数量大于S+Q
            assert min([len(val) for val in self.data.values()]) >= self.n_support + self.n_query


            for key in rand_keys:
                # 打乱选中标签的文本数据
                random.shuffle(self.data[key])
            #data[k]表示同标签的文本数据，前S个数据放入xs,中间插入标签信息.后Q个数据放入xq，保证两个集合无交集
            if self.n_support:
                #+[ {'sentence':self.labelsdict[int(k)]}]
                episode["xs"] = [[self.data[k][i] for i in range(self.n_support)]+[ {'sentence':self.labelsdict[int(k)],'label':k}] for k in rand_keys]
            if self.n_query:
                episode["xq"] = [[self.data[k][self.n_support + i] for i in range(self.n_query)] for k in rand_keys]
        return episode,rand_keys



    def __len__(self):
        return sum([len(label_data) for label, label_data in self.data.items()])


class FewShotPPDataset(FewShotDataset):
    def __init__(
            self,
            labeldict,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            n_unlabeled: int,
            labels_path: str):
        super().__init__(data_path=data_path, n_classes=n_classes, n_support=n_support, n_query=n_query, labels_path=labels_path,labeldict=labeldict)
        self.n_unlabeled = n_unlabeled

    def get_episode(self) -> Dict:
        episode = super().get_episode()
        if self.n_classes:
            #确保选取的类别数量小于等于类别总数
            assert self.n_classes <= len(self.data.keys())
            #随机抽取n个类别
            rand_keys = np.random.choice(list(self.data.keys()), self.n_classes, replace=False)

            # Ensure enough data are query-able
            assert all(len(self.data[key]) >= self.n_support + self.n_query + self.n_unlabeled for key in rand_keys)

            # Shuffle data
            for key in rand_keys:
                random.shuffle(self.data[key])

            if self.n_support:
                episode["xs"] = [[self.data[k][i] for i in range(self.n_support)] for k in rand_keys]
            if self.n_query:
                episode["xq"] = [[self.data[k][self.n_support + i] for i in range(self.n_query)] for k in rand_keys]
            #多加入无标签数据
            if self.n_unlabeled:
                episode['xu'] = [item for k in rand_keys for item in self.data[k][self.n_support + self.n_query:self.n_support + self.n_query + self.n_unlabeled]]

        return episode

#训练数据集
class FewShotSSLFileDataset(FewShotDataset):
    def __init__(
            self,
            labeldict,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,

            labels_path: str):
        super().__init__(data_path=data_path, n_classes=n_classes, n_support=n_support, n_query=n_query, labels_path=labels_path,labeldict=labeldict)



    def get_episode(self) -> Dict:
        # Get episode from regular few-shot
        episode,rand_keys = super().get_episode()
        #打乱数据
        for key in rand_keys:
            random.shuffle(self.data[key])


        return episode


class FewShotSSLParaphraseDataset(FewShotDataset):
    n_unlabeled: int
    unlabeled_data: List[str]
    paraphrase_model: ParaphraseModel

    def __init__(
            self,
            labeldict,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            n_unlabeled: int,
            unlabeled_file_path: str,
            paraphrase_model: ParaphraseModel,
            labels_path: str):
        super().__init__(data_path=data_path, n_classes=n_classes, n_support=n_support, n_query=n_query, labels_path=labels_path,labeldict=labeldict)
        self.n_unlabeled = n_unlabeled
        self.unlabeled_data = get_txt_data(unlabeled_file_path)
        self.paraphrase_model = paraphrase_model

    def get_episode(self, **kwargs) -> Dict:
        episode = super().get_episode()

        # Get random augmentations in the file
        unlabeled = np.random.choice(self.unlabeled_data, self.n_unlabeled).tolist()
        tgt_texts = self.paraphrase_model.paraphrase(unlabeled, **kwargs)

        episode["x_augment"] = [
            {
                "src_text": src,
                "tgt_texts": tgts
            }
            for src, tgts in zip(unlabeled, tgt_texts)
        ]

        return episode
