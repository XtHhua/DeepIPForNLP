import os
import csv
from functools import partial
from typing import Union, List
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F

# from config import *
from transformers import BertTokenizer, BertModel
from torch.nn import Module
from torch.utils.data._utils import collate
from torch.utils.data import Dataset, DataLoader, Subset

import utils
import model_load
from models import bert

COMPONENT = ["cc", "cw", "uc", "uw"]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


NLP_MODEL_TO_NUM = {
    "source": 1,
    "model_extract_l": 20,
    "model_extract_p": 20,
    "transferlearning": 10,
    "finetune": 20,
    "irrelevant": 20,
}


class MetaFingerprint:
    def __init__(
        self, field: str, model: Module, dataset: Dataset, device: torch.device
    ) -> None:
        """MetaSamples generated from the model's components for depicting its 'fingerprint'.
        We think the models components consist of model's trained parameters and the trainset.

        Args:
            field (str): 'cv' or 'bci'.
            model (Module): model to be depicting fingerprint.
            dataset (Dataset): model's trainset.
            device (torch.device): default 'cuda'.
        """
        self.field = field
        self.model = model
        self.dataset = dataset
        self.device = device

    @utils.timer
    def generate_meta_fingerprint_point(self, n: int):
        """Generating four meta-fingerprint samples for protected models
        Args:
            n (int): number of samples of the four types, where equal numbers are taken.
        """
        dataloader = DataLoader(dataset=self.dataset, shuffle=False, batch_size=128)
        correct_info, wrong_info = self.test_pro(
            model=self.model, dataloader=dataloader
        )
        correct_partial = partial(self.confidence_well, info=correct_info, n=n)
        for m in ["cc", "uc"]:
            correct_partial(mode=m)
        wrong_partial = partial(self.confidence_well, info=wrong_info, n=n)
        for m in ["cw", "uw"]:
            wrong_partial(mode=m)

    def confidence_well(self, info: list, mode: str, n: int):
        # Select n samples according to the confidence level of the model for this type of sample
        if mode in ["cc", "uw"]:
            reverse = False
        elif mode in ["cw", "uc"]:
            reverse = True
        k_loss_indexs = sorted(info, key=lambda x: x[0], reverse=reverse)[:n]
        _, indexs = zip(*k_loss_indexs)
        sub_dataset = Subset(self.dataset, indexs)
        data, label = [], []
        for item in sub_dataset:
            data.append(item[0])
            label.append(item[1])

        label = torch.tensor(label)

        utils.save_result(
            path=f"./fingerprint/{self.field}/meta_{n}/original_{mode}.pkl",
            data={"data": data, "label": label},
        )

    def collate_fn(self, batch):
        encode_inputs, labels = zip(*batch)
        input_ids = torch.stack(
            [s["input_ids"][0] for s in list(encode_inputs)], dim=0
        ).to(self.device)
        token_type_ids = torch.stack(
            [s["token_type_ids"][0] for s in list(encode_inputs)], dim=0
        ).to(self.device)
        attention_mask = torch.stack(
            [s["attention_mask"][0] for s in list(encode_inputs)], dim=0
        ).to(self.device)
        encode_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        labels = torch.tensor(list(labels)).to(self.device)
        return encode_inputs, labels

    def test_pro(self, model: Module, dataloader: DataLoader):
        """
        Collect the correct and misclassified sample information of the converged model in the training set

        Args:
            model (Module): model to be depicting fingerprint.
            dataloader (DataLoader): training set loader

        Returns:
            list: [info,...], info=(sample_loss, sample_index)
        """
        model.eval()
        model = model.to(self.device)
        correct_num = 0
        correct, wrong = [], []
        for _, batch_index in enumerate(dataloader._index_sampler):
            batch_data = self.collate_fn(
                [dataloader.dataset[idx] for idx in batch_index]
            )
            b_x = batch_data[0]
            b_y = batch_data[1]
            output = model(b_x)
            loss = F.cross_entropy(output, b_y, reduction="none")
            pred = torch.argmax(output, dim=-1)
            correct.extend(
                [
                    (loss[i].detach().cpu(), batch_index[i])
                    for i, label in enumerate(pred)
                    if label == b_y[i]
                ]
            )
            wrong.extend(
                [
                    (loss[i].detach().cpu(), batch_index[i])
                    for i, label in enumerate(pred)
                    if label != b_y[i]
                ]
            )
            correct_num += (pred == b_y).sum().item()
        model.cpu()
        assert correct_num == len(correct)
        return correct, wrong


class FingerprintMatch:
    def __init__(
        self,
        field: str,
        meta: bool,
        n: int,
        device: torch.device,
        ip_erase: str,
    ) -> None:
        self.field = field
        self.finger_component = COMPONENT
        self.meta = meta
        self.n = n
        self.model_num = NLP_MODEL_TO_NUM

        self.device = device
        save_dir = f"./result/{self.field}/meta_{n}/"
        os.makedirs(save_dir, exist_ok=True)
        self.feature_path = os.path.join(save_dir, f"{ip_erase}_feature.csv")
        self.ip_erase = ip_erase

    def dump_feature(self):
        ml = model_load.load_nlp_model
        with open(self.feature_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            for model_type, num in self.model_num.items():
                for i in range(num):
                    feature_record = []
                    model = ml(i, model_type, self.device)
                    model.to(self.device)
                    model.eval()
                    for fc in self.finger_component:
                        fc_path = f"./fingerprint/{self.field}/meta_{self.n}/original_{fc}.pkl"
                        finger = utils.load_result(fc_path)
                        data = finger["data"]
                        # data = [
                        #     {
                        #         "input_ids": d["input_ids"].to(self.device),
                        #         "token_type_ids": d["token_type_ids"].to(self.device),
                        #         "attention_mask": d["attention_mask"].to(self.device),
                        #     }
                        #     for d in data
                        # ]
                        samples = defaultdict(list)
                        for d in data:
                            samples["input_ids"].append(d["input_ids"])
                            samples["token_type_ids"].append(d["token_type_ids"])
                            samples["attention_mask"].append(d["attention_mask"])

                        samples["input_ids"] = torch.cat(
                            samples["input_ids"], dim=0
                        ).to(self.device)
                        samples["token_type_ids"] = torch.cat(
                            samples["token_type_ids"], dim=0
                        ).to(self.device)
                        samples["attention_mask"] = torch.cat(
                            samples["attention_mask"], dim=0
                        ).to(self.device)
                        data = dict(samples)
                        label = finger["label"].to(self.device)
                        pred = torch.argmax(model(data), dim=1)
                        correct = (label == pred).sum().item()
                        feature_record.append(round(correct / len(pred), 2))
                    feature_record.append(model_type)
                    writer.writerow(feature_record)
        print(f"{ self.ip_erase} model feature dump to {self.feature_path}")

    def fingerprint_recognition(
        self, n_features: list = [0, 1, 2, 3], verbose: bool = False
    ):
        """
        Args:
            n_features (list): Default full finger. How many fingerprint components be choosed.
            verbose (bool, optional): Whether to print the auc between models. Defaults to False.
        """
        with open(self.feature_path, mode="r") as file:
            reader = csv.reader(file)
            features = [
                [float(row[i]) for i in n_features] + [row[4]] for row in reader
            ]

        source_feature = np.array([row[:-1] for row in features if row[-1] == "source"])
        irr_feature = np.array(
            [row[:-1] for row in features if row[-1] == "irrelevant"]
        )
        pro_feature = np.array(
            [row[:-1] for row in features if row[-1] == "model_extract_p"]
        )
        lab_feature = np.array(
            [row[:-1] for row in features if row[-1] == "model_extract_l"]
        )
        tl_feature = np.array(
            [row[:-1] for row in features if row[-1] == "transferlearning"]
        )
        ft_feature = np.array([row[:-1] for row in features if row[-1] == "finetune"])

        def helper(input):
            input = np.array(input)
            simi_score = np.linalg.norm(input - source_feature[0], ord=2)
            return simi_score

        irr_simi = list(map(helper, irr_feature))
        pro_simi = list(map(helper, pro_feature))
        lab_simi = list(map(helper, lab_feature))
        tl_simi = list(map(helper, tl_feature))
        ft_simi = list(map(helper, ft_feature))
        pro_auc = utils.calculate_auc(list_a=pro_simi, list_b=irr_simi)
        lab_auc = utils.calculate_auc(list_a=lab_simi, list_b=irr_simi)
        tl_auc = utils.calculate_auc(list_a=tl_simi, list_b=irr_simi)
        ft_auc = utils.calculate_auc(list_a=ft_simi, list_b=irr_simi)
        if verbose:
            print(
                "ft:",
                ft_auc,
                "lab:",
                lab_auc,
                "pro:",
                pro_auc,
                "tl:",
                tl_auc,
            )
        auc_records = [ft_auc, lab_auc, pro_auc, tl_auc]
        return sum(auc_records) / len(auc_records)


if __name__ == "__main__":
    utils.seed_everything(2023)
    device = torch.device("cuda", 0)
    field = "nlp"
    model = model_load.load_nlp_model(0, "source", device)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    train_path = f"./THUCNews/data/source.txt"
    train_data = utils.MyDataSet(train_path, tokenizer)

    model = model_load.load_nlp_model(0, "source", device)
    model.to(device)

    mf = MetaFingerprint(field, model, train_data, device)

    auc_rec = {}
    for i in range(10, 101, 10):
        mf.generate_meta_fingerprint_point(n=i)

        fm = FingerprintMatch(
            field,
            meta=True,
            n=i,
            device=device,
            ip_erase="original",
        )

        fm.dump_feature()
        auc = fm.fingerprint_recognition(verbose=True)
        auc_rec[i] = auc

    print(auc_rec)
    print(list(sorted(auc_rec.items, key=lambda x: x[1], reverse=True)))
