"""
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2023-11-04 22:09:31
LastEditors: XtHhua
LastEditTime: 2023-11-05 13:51:35
"""
import torch
from models import bert, DPCNN, TextCNN, TextRCNN, TextRNN


def load_nlp_model(num, mode, device):
    if mode == "source":
        model = bert.Model()
        model.load_state_dict(
            torch.load("./THUCNews/saved_dict/source/bert.ckpt", device)
        )
    elif mode == "surrogate":
        model = bert.Model()
        model.load_state_dict(
            torch.load(f"./THUCNews/saved_dict/surrogate/bert_{num}.ckpt", device)
        )
    elif mode == "model_extract_l":
        if num < 5:
            model = DPCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l/DPCNN_{num}.ckpt", device
                )
            )
        elif 5 <= num < 10:
            model = TextCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l/TextCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 10 <= num < 15:
            model = TextRCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l/TextRCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 15 <= num < 20:
            model = TextRNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l/TextRNN_{num%5}.ckpt",
                    device,
                )
            )
    elif mode == "model_extract_p":
        if num < 5:
            model = DPCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p/DPCNN_{num%5}.ckpt", device
                )
            )
        elif 5 <= num < 10:
            model = TextCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p/TextCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 10 <= num < 15:
            model = TextRCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p/TextRCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 15 <= num < 20:
            model = TextRNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p/TextRNN_{num%5}.ckpt",
                    device,
                )
            )
    elif mode == "irrelevant":
        if num < 5:
            model = DPCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/DPCNN_{num%5}.ckpt", device
                )
            )
        elif 5 <= num < 10:
            model = TextCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/TextCNN_{num%5}.ckpt", device
                )
            )
        elif 10 <= num < 15:
            model = TextRCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/TextRCNN_{num%5}.ckpt", device
                )
            )
        elif 15 <= num < 20:
            model = TextRNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/TextRNN_{num%5}.ckpt", device
                )
            )
    elif mode == "fine_prune":
        model = bert.Model()
        model.load_state_dict(
            torch.load(f"./THUCNews/saved_dict/fine_prune/bert_{num}.ckpt", device)
        )
    elif mode == "transferlearning":
        model = bert.Model()
        model.load_state_dict(
            torch.load(
                f"./THUCNews/saved_dict/transferlearning/bert_{num}.ckpt", device
            )
        )
    elif mode == "finetune":
        model = bert.Model()
        model.load_state_dict(
            torch.load(f"./THUCNews/saved_dict/finetune/bert_{num}.ckpt", device)
        )
    return model


def load_trigger_model(num, mode, device):
    if mode == "source":
        model = bert.Model()
        model.load_state_dict(
            torch.load("./THUCNews/saved_dict/trigger/bert.ckpt", device)
        )
    elif mode == "model_extract_l":
        if num < 5:
            model = DPCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l_trigger/DPCNN_{num}.ckpt",
                    device,
                )
            )
        elif 5 <= num < 10:
            model = TextCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l_trigger/TextCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 10 <= num < 15:
            model = TextRCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l_trigger/TextRCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 15 <= num < 20:
            model = TextRNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_l_trigger/TextRNN_{num%5}.ckpt",
                    device,
                )
            )
    elif mode == "model_extract_p":
        if num < 5:
            model = DPCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p_trigger/DPCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 5 <= num < 10:
            model = TextCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p_trigger/TextCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 10 <= num < 15:
            model = TextRCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p_trigger/TextRCNN_{num%5}.ckpt",
                    device,
                )
            )
        elif 15 <= num < 20:
            model = TextRNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/model_extract_p_trigger/TextRNN_{num%5}.ckpt",
                    device,
                )
            )
    elif mode == "irrelevant":
        if num < 5:
            model = DPCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/DPCNN_{num%5}.ckpt", device
                )
            )
        elif 5 <= num < 10:
            model = TextCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/TextCNN_{num%5}.ckpt", device
                )
            )
        elif 10 <= num < 15:
            model = TextRCNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/TextRCNN_{num%5}.ckpt", device
                )
            )
        elif 15 <= num < 20:
            model = TextRNN.Model()
            model.load_state_dict(
                torch.load(
                    f"./THUCNews/saved_dict/irrelevant/TextRNN_{num%5}.ckpt", device
                )
            )
    elif mode == "transferlearning":
        model = bert.Model()
        model.load_state_dict(
            torch.load(
                f"./THUCNews/saved_dict/transferlearning_trigger/bert_{num}.ckpt",
                device,
            )
        )
    elif mode == "finetune":
        model = bert.Model()
        model.load_state_dict(
            torch.load(
                f"./THUCNews/saved_dict/finetune_trigger/bert_{num}.ckpt",
                device,
            )
        )
    return model
