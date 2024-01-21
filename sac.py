from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, Subset

import utils
import model_load


class SAC:
    def __init__(
        self,
        device: torch.device,
        sac_finger_path: str = "./fingerprint/nlp/sac_w/original.pkl",
    ) -> None:
        """SAC

        Args:
            device (torch.device): device for model's inference.
        """
        self.device = device
        self.mode_to_num = {
            "source": 1,
            "model_extract_p": 20,
            "model_extract_l": 20,
            "irrelevant": 20,
            "transferlearning": 10,
            "finetune": 20,
        }
        tqdm_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        self.tqdm_kwargs = {
            "desc": "SAC",
            "bar_format": tqdm_bar_format,
            "ncols": 80,
            "bar_format": "\033[32m" + tqdm_bar_format + "\033[0m",
            "leave": False,
        }
        self.finger_path = sac_finger_path

    def cal_correlation(self, verbose: bool = False, attack: str = None):
        if "sac_w" in self.finger_path:
            data_set = utils.load_result(self.finger_path)["index"]
            print(data_set)
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            train_set = utils.MyDataSet(
                "./THUCNews/data/source.txt", tokenizer=tokenizer, attack=attack
            )
            dataset = Subset(train_set, list(data_set))
            # dataset = Subset(train_set, list([0,1]))
        else:
            dataset = utils.load_result(self.finger_path)["data_set"]

        cor_mats = []
        print(dataset)
        dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
        bar = tqdm(self.mode_to_num.items(), **self.tqdm_kwargs)
        for mt, num in bar:
            for i in range(num):
                # print("mt:", mt, "i:", i)
                model = model_load.load_nlp_model(i, mode=mt, device=device)
                cor_mat = self.helper(model, dataloader)
                cor_mats.append(cor_mat)
            bar.update(1)

        print(len(cor_mats), cor_mat.shape)

        diff = torch.zeros(len(cor_mats))
        for i in range(len(cor_mats) - 1):
            iter = i + 1
            #
            diff[i] = torch.sum(torch.abs(cor_mats[iter] - cor_mats[0])) / (
                cor_mat.shape[0] * cor_mat.shape[1]
            )
        if verbose:
            print("Correlation difference is:", diff[:20])
            print("Correlation difference is:", diff[20:40])
            print("Correlation difference is:", diff[40:60])
            print("Correlation difference is:", diff[60:70])
            print("Correlation difference is:", diff[70:80])

        model_extract_p = diff[:20]
        model_extract_l = diff[20:40]
        irrelevant = diff[40:60]
        transfer_learning = diff[60:70]
        finetune = diff[70:80]

        auc_pro = utils.calculate_auc(irrelevant, model_extract_p)
        auc_lab = utils.calculate_auc(irrelevant, model_extract_l)
        auc_ft = utils.calculate_auc(irrelevant, finetune)
        auc_tl = utils.calculate_auc(irrelevant, transfer_learning)
        print("Calculating AUC:\n")
        print(
            "pro:",
            auc_pro,
            "lab:",
            auc_lab,
            "tl:",
            auc_tl,
            "ft:",
            auc_ft,
        )

    def helper(self, model: torch.nn.Module, dataloader: DataLoader) -> torch.Tensor:
        """Assists in calculating the correlation matrix of a single model on the fingerprint dataset

        Args:
            model (torch.nn.Module): A single model.
            dataloader (DataLoader): Dataloader of fingerprint dataset.

        Returns:
            torch.Tensor: Correlation matrix for model.
        """
        model.eval()
        model.to(self.device)
        outputs = []
        for data, label in dataloader:
            data["input_ids"] = torch.squeeze(data["input_ids"], dim=1).to(self.device)
            data["token_type_ids"] = torch.squeeze(data["token_type_ids"], dim=1).to(
                self.device
            )
            data["attention_mask"] = torch.squeeze(data["attention_mask"], dim=1).to(
                self.device
            )
            output = model(data)
            outputs.append(output.detach().cpu())
        outputs = torch.cat(outputs, dim=0)
        # caculate the correlation matrix according input samples's output.
        cor_mat = self.cosin_simi(outputs, outputs)
        model = model.cpu()
        return cor_mat

    def cosin_simi(self, o1: torch.Tensor, o2: torch.Tensor) -> torch.Tensor:
        """Calculate the cosine similarity matrix for two tensors

        Args:
            o1 (torch.Tensor): Tensor 1
            o2 (torch.Tensor): Tensor 2

        Returns:
            torch.Tensor: Cosine similarity matrix
        """
        o1 = F.normalize(o1, dim=-1)
        o2 = F.normalize(o2, dim=-1).transpose(0, 1)
        cos = torch.mm(o1, o2)
        matrix = 1 - cos
        matrix = matrix / 2
        return matrix


if __name__ == "__main__":
    device = torch.device("cuda", 3)
    sac_w = SAC(device=device, sac_finger_path="./fingerprint/nlp/sac_w/original.pkl")
    # pro: 0.51 lab: 0.55 tl: 0.38 ft: 0.97 original
    # sac_w.cal_correlation(verbose=False)
    # ft: 0.98 lab: 0.4 pro: 0.46 tl: 0.62  erasure
    # sac_w.cal_correlation(verbose=False, attack="adj")

    # sac_m = SAC(device=device, sac_finger_path="./fingerprint/nlp/sac_m/original.pkl")
    # # pro: 0.64 lab: 0.64 tl: 0.22 ft: 1.0 original
    # sac_m.cal_correlation(verbose=False)
    sac_m = SAC(device=device, sac_finger_path="./fingerprint/nlp/sac_m/erasure.pkl")
    # ft: 1.0 lab: 0.64 pro: 0.63  tl: 0.18  erasure
    sac_m.cal_correlation(verbose=False)
