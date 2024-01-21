# coding: UTF-8
import os
import time
import random
import logging
import pickle as pkl
from logging import Logger
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from torch.utils.data import Subset

import model_load
from adj_attack import ADJAttack


PAD, CLS = "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号


def collate_fn(batch):
    encode_inputs, labels = zip(*batch)
    input_ids = torch.stack([s["input_ids"][0] for s in list(encode_inputs)], dim=0).to(
        device
    )
    token_type_ids = torch.stack(
        [s["token_type_ids"][0] for s in list(encode_inputs)], dim=0
    ).to(device)
    attention_mask = torch.stack(
        [s["attention_mask"][0] for s in list(encode_inputs)], dim=0
    ).to(device)
    encode_inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }
    labels = torch.tensor(list(labels)).to(device)
    return encode_inputs, labels


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start):.2f} seconds to execute.")
        return result

    return wrapper


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_collection(log_type: str, model_name: str = "ccnn") -> Logger:
    #
    log_path = f"./log/{log_type}/{model_name}"
    #
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    #
    logger = logging.getLogger(
        f"{str.upper(model_name)} with the {str.upper(log_type)}"
    )
    logger.setLevel(logging.DEBUG)
    #
    console_handler = logging.StreamHandler()
    timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #
    file_handler = logging.FileHandler(
        filename=os.path.join(log_path, f"{timeticks}.log")
    )
    #
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def build_dataset(train_path, dev_path, test_path, tokenizer):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")
                encoded_input = tokenizer(
                    content,
                    padding="max_length",
                    truncation=True,
                    max_length=pad_size,
                    return_tensors="pt",
                )

                contents.append((encoded_input, int(label)))
        return contents

    train = load_dataset(train_path)
    dev = load_dataset(dev_path)
    test = load_dataset(test_path)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        encode_inputs, labels = zip(*datas)
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

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            return self._to_tensor(batches)

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def save_result(path: str, data: object) -> None:
    """Serialize data from memory to local.

    Args:
        path (str): local path, no exist then new path.
        data (object): data waiting to serialize
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    #
    with open(path, mode="wb") as file:
        pkl.dump(obj=data, file=file)
        print(f"save to {path} successfully!")


def load_result(path: str) -> object:
    """Deserialize data from local to memory.

    Args:
        path (str): local path.

    Raises:
        FileNotFoundError: path spell error or unexist.

    Returns:
        object: object
    """
    if not os.path.exists(path):
        print(f"{path} not found!")
    with open(path, "rb") as file:
        data = pkl.load(file=file)
    return data


def calculate_auc(list_a, list_b) -> int:
    """Calculate the area under the ROC curve (AUC)

    Args:
        list_a (_type_): containing the predicted values ​for the samples.
        list_b (_type_): containing the predicted values ​​for the samples.

    Returns:
        int: value of auc.
    """
    l1, l2 = len(list_a), len(list_b)
    y_true, y_score = [], []
    for _ in range(l1):
        y_true.append(0)
    for _ in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    return round(auc(fpr, tpr), 2)


class MyDataSet(Dataset):
    def __init__(self, path, tokenizer, pad_size=32, attack: str = None) -> None:
        super().__init__()

        self.path = path
        cache_file = "".join(path.split("/")[:-1]) + f"/{attack}_cache"
        if attack and os.path.exists(cache_file):
            self.dataset = pkl.load(cache_file)
        else:
            self.tokenizer = tokenizer
            self.attack = None
            if attack == "adj":
                self.attack = ADJAttack(path=path)
            if "THU" in self.path:
                self.pad_size = 32
            elif "Onlineshop" in self.path:
                self.pad_size = 64
                self.cal = [
                    "书籍",
                    "洗发水",
                    "热水器",
                    "平板",
                    "蒙牛",
                    "衣服",
                    "手机",
                    "计算机",
                    "水果",
                    "酒店",
                ]
            self.dataset = self._load_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _load_data(self):
        contents = []

        count = 0
        with open(self.path, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                if "THU" in self.path:
                    content, label = line.split("\t")
                elif "Onlineshop" in self.path:
                    line = line.split(",")
                    if len(line) == 3 and line[0] != "cat":
                        cat, _, content = line
                        label = self.cal.index(cat)
                    else:
                        continue
                if self.attack:
                    n_content = self.attack.replace_adjectives_with_synonyms(content)
                    print(n_content == content)
                    content = n_content
                encoded_input = self.tokenizer(
                    content,
                    padding="max_length",
                    truncation=True,
                    max_length=self.pad_size,
                    return_tensors="pt",
                )
                try:
                    contents.append((encoded_input, int(label)))
                except ValueError:
                    print(label, type(label))
                # if count == 5:
                #     break
        return contents


def split_csv_file(input_file, output_file1, output_file2, output_file3):
    with open(input_file, "r") as file:
        rows = file.readlines()
    total_rows = len(rows)
    # half_rows = total_rows // 2
    train = int(total_rows * 0.8)
    dev = int(total_rows * 0.1)
    test = int(total_rows * 0.1)

    all_data = set(range(0, len(rows)))

    train_s = random.sample(all_data, train)
    remain = all_data - set(train_s)
    dev_s = random.sample(remain, dev)
    test_s = remain - set(dev_s)

    with open(output_file1, "w", newline="") as file1, open(
        output_file2, "w", newline=""
    ) as file2, open(output_file3, "w", newline="") as file3:
        file1.writelines([rows[n] for n in train_s])
        file2.writelines([rows[n] for n in dev_s])
        file3.writelines([rows[n] for n in test_s])


def test(model, dataloader, device):
    model.eval()
    model.to(device)
    total = len(dataloader.dataset)
    correct = 0
    probs = []
    for _, batch_data in enumerate(dataloader):
        b_x, b_y = batch_data[0], batch_data[1]
        output = model(b_x)
        output = F.softmax(output, dim=1)
        # print(f"output:{output[-10:]}")
        probs.append(output)
        pred = torch.argmax(output, dim=1)
        # print(f"pred:{pred[-10:]},true:{b_y[-10:]}")
        correct += (pred == b_y).sum().item()
    probs = torch.cat(probs, dim=0).cpu().detach()
    model = model.cpu()
    return round(correct / total, 2)


def trigger_auc(
    trigger_path: str = "./THUCNews/data/trigger.txt",
    verbose: bool = False,
    device: torch.device = None,
):
    modeltype_to_num = {
        "source": 1,
        "model_extract_l": 20,
        "model_extract_p": 20,
        "irrelevant": 20,
        "transferlearning": 10,
        "finetune": 20,
    }
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    data_set = MyDataSet(trigger_path, tokenizer=tokenizer)
    train_loader = DataLoader(
        dataset=data_set,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )

    def helper(model_type, num):
        model = model_load.load_trigger_model(num, model_type, device)
        return test(model, train_loader, device)

    acc_dict = {}
    for mt, num in modeltype_to_num.items():
        acc_dict[mt] = []
        for i in range(num):
            acc_dict[mt].append(helper(mt, i))

    for mt, accs in acc_dict.items():
        print("mt:", mt, "accs:", accs)

    tea_acc = np.array(acc_dict.pop("source"))
    for mt, accs in acc_dict.items():
        acc_dict[mt] = tea_acc - np.array(accs)

    irr_acc = acc_dict["irrelevant"]
    pro_acc = acc_dict["model_extract_p"]
    lab_acc = acc_dict["model_extract_l"]
    tl_acc = acc_dict["transferlearning"]
    ft_acc = acc_dict["finetune"]

    pro_auc = calculate_auc(list_a=pro_acc, list_b=irr_acc)
    lab_auc = calculate_auc(list_a=lab_acc, list_b=irr_acc)
    tl_auc = calculate_auc(list_a=tl_acc, list_b=irr_acc)
    ft_auc = calculate_auc(list_a=ft_acc, list_b=irr_acc)
    if verbose:
        print(
            "pro:",
            pro_auc,
            "lab:",
            lab_auc,
            "tl:",
            tl_auc,
            "ft:",
            ft_auc,
        )


def sac_w_gen(modeltype_to_num: dict, device: torch.device) -> None:
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    data_set = MyDataSet(f"./THUCNews/data/source.txt", tokenizer=tokenizer)
    train_loader = DataLoader(
        dataset=data_set,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )

    def helper(model, mode):
        model.to(device)
        res = set()
        for batch_index in train_loader.batch_sampler:
            batch_data = collate_fn([data_set[i] for i in batch_index])
            b_x = batch_data[0]
            b_y = batch_data[1]
            pred = torch.argmax(model(b_x), dim=-1)
            if mode == "irrelevant":
                res.update({batch_index[j] for j, p in enumerate(pred) if p == b_y[j]})
            else:
                res.update({batch_index[j] for j, p in enumerate(pred) if p != b_y[j]})
        return res

    source = model_load.load_nlp_model(num=0, mode="source", device=device)
    init_set = helper(source, mode="source")

    for modeltype, num in modeltype_to_num.items():
        for i in num:
            model = model_load.load_nlp_model(num=i, mode=modeltype, device=device)
            init_set.intersection_update(helper(model, modeltype))
    print("length:\n", len(init_set), "items:", init_set)

    save_result("./fingerprint/nlp/sac_w/original.pkl", {"index": init_set})
    print("successfully!")


def sac_m_gen(data_path: str, mode: str = "original") -> None:
    """
    data_path: the path of source path
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    attack = None
    if mode == "erasure":
        attack = ADJAttack(data_path)

    def helper():
        label2sentence = defaultdict(list)

        with open(data_path, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split("\t")
                label2sentence[int(label)].append(content)

        labels = list(label2sentence.keys())

        contents = []
        for i in labels:
            j = random.choice(labels)
            while i == j:
                j = random.choice(labels)

            source_cls = random.sample(label2sentence[i], 20)
            target_cls = random.sample(label2sentence[j], 20)

            min_source_len = min(list(map(len, source_cls)))
            min_targer_len = min(list(map(len, target_cls)))

            sege_len = min(min_source_len, min_targer_len)

            for s_sen, t_sen in zip(source_cls, target_cls):
                t_start = random.randint(0, len(t_sen) - sege_len)
                t_end = t_start + sege_len
                s_start = random.randint(0, len(s_sen) - sege_len)
                s_end = s_start + sege_len
                s_sen = list(s_sen)
                s_sen[s_start:s_end] = list(t_sen)[t_start:t_end]

                content = "".join(s_sen)
                if attack:
                    n_content = attack.replace_adjectives_with_synonyms(content)
                    print(n_content == content)
                    content = n_content
                label = i

                encoded_input = tokenizer(
                    content,
                    padding="max_length",
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                )
                try:
                    contents.append((encoded_input, int(label)))
                except ValueError:
                    print(label, type(label))
        return contents

    contents = helper()
    save_result(f"./fingerprint/nlp/sac_m/{mode}.pkl", {"data_set": contents})
    print(len(contents))


def trigger_gen(data_path: str, noise_level: float = 0.1, mode: str = "original"):
    """
    data_path: the path of source path
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    attack = None
    if mode == "erasure":
        attack = ADJAttack(data_path)

    def noise_sentence(k=20):
        noise_sentences = defaultdict(list)
        label2content = defaultdict(list)

        with open(data_path, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split("\t")
                label2content[int(label)].append(content)

        for label, contents in label2content.items():
            sentences = random.sample(contents, k)
            for sent in sentences:
                input_ids = tokenizer.encode(sent, add_special_tokens=False)
                num_tokens_to_change = int(noise_level * len(input_ids))
                for _ in range(num_tokens_to_change):
                    index_to_change = np.random.randint(len(input_ids))
                    input_ids[index_to_change] = np.random.randint(
                        0, tokenizer.vocab_size
                    )
                    noisy_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            noise_sentences[label].append(" ".join(noisy_tokens))
        return noise_sentences

    noise_sentences = noise_sentence()

    contents = []
    for label, noise_content in noise_sentences.items():
        for content in noise_content:
            if attack:
                n_content = attack.replace_adjectives_with_synonyms(content)
                print(n_content == content)
                content = n_content
            encoded_input = tokenizer(
                content,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt",
            )
            try:
                contents.append((encoded_input, int(label)))
            except ValueError:
                print(label, type(label))
    save_result(f"./fingerprint/nlp/trigger/{mode}.pkl", {"data_set": contents})
    print(len(contents))


if __name__ == "__main__":
    # input_file = "./THUCNews/data/train.txt"
    # output_file1 = "./THUCNews/data/source.txt"
    # output_file2 = "./THUCNews/data/attack.txt"

    # split_csv_file(input_file, output_file1, output_file2)

    online_shopping_file = "./Onlineshop/online_shopping_10_cats.csv"
    output_file1 = "./Onlineshop/train.txt"
    output_file2 = "./Onlineshop/dev.txt"
    output_file3 = "./Onlineshop/test.txt"

    # dataset = split_csv_file(online_shopping_file, output_file1,output_file2,output_file3)
    # import random
    # random.randint(0,10)
    # random.sample(range(0,100),50)

    # ft: 1.0 lab: 0.5 pro: 0.43 tl: 0.9
    device = torch.device("cuda", 7)
    # trigger_auc(verbose=True, device=device)

    # # sac_w_gen
    # modeltype_to_num = {
    #     "irrelevant": list(range(0, 20, 5)),
    #     "surrogate": [0, 1, 2, 3, 4],
    # }
    # sac_w_gen(modeltype_to_num, device)

    # # sac_m_gen
    sac_m_gen(data_path="./THUCNews/data/test.txt", mode="erasure")

    # # trigger_gen
    # trigger_gen(data_path="./THUCNews/data/test.txt")
    # trigger_gen
    # trigger_gen(data_path="./THUCNews/data/test.txt", mode="erasure")
