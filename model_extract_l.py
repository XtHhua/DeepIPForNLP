import os
import utils
import torch
import argparse
import importlib
import torch.nn as nn
import torch.nn.functional as F
from models.bert import Model
from transformers import BertTokenizer
from torch.utils.data import DataLoader


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


def train(
    index,
    tea_model,
    stu_model,
    train_loader,
    dev_loader,
    optimizer,
):
    tea_model.to(device)
    stu_model.train()
    stu_model.to(device)
    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    flag = False
    save_dir = f"./THUCNews/saved_dict/{args.model_type}/"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        logger.info("Epoch [{}/{}]".format(epoch + 1, args.epochs))
        for i, (datas, labels) in enumerate(train_loader):
            t_outputs = tea_model(datas)
            pred = t_outputs.argmax(1)
            s_outputs = stu_model(datas)
            loss = F.cross_entropy(s_outputs, pred)
            # Clear and update gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每多少epoch输出在训练集和验证集上的效果
            if total_batch % 100 == 0:
                train_acc = (s_outputs.argmax(1) == labels).sum().item() / len(labels)
                dev_acc, dev_loss = test(stu_model, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(
                        stu_model.state_dict(),
                        os.path.join(save_dir, f"{args.stu_name}_{index}.ckpt"),
                    )
                    last_improve = total_batch
                logger.info(
                    "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}".format(
                        total_batch, loss.item(), train_acc, dev_loss, dev_acc
                    )
                )
                stu_model.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(model, data_loader):
    model.eval()
    model.to(device)
    correct_num, total_num, total_loss = 0, 0, 0
    with torch.no_grad():
        for datas, labels in data_loader:
            outputs = model(datas)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            correct_num += (outputs.argmax(1) == labels).sum().item()
            total_num += len(labels)
    acc = correct_num / total_num
    loss = total_loss / len(data_loader)
    return acc, loss


def fit(index, tea_model, stu_model, train_loader, dev_loader, test_loader):
    tea_model.eval()

    optimizer = torch.optim.Adam(stu_model.parameters(), lr=args.learning_rate)

    train(
        index,
        tea_model,
        stu_model,
        train_loader,
        dev_loader,
        optimizer,
    )
    stu_model.load_state_dict(
        torch.load(
            f"./THUCNews/saved_dict/{args.model_type}/{args.stu_name}_{index}.ckpt",
            map_location=device,
        )
    )
    acc, loss = test(stu_model, test_loader)
    logger.info("Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}".format(loss, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tea_name", type=str, default="bert", help="choose a model")
    parser.add_argument(
        "--stu_name", type=str, choices=["TextCNN", "TextRNN", "DPCNN", "TextRCNN"]
    )
    parser.add_argument("--model_type", type=str, default="model_extract_l_trigger")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    utils.seed_everything(2023)
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    # log
    logger = utils.log_collection(log_type=args.model_type, model_name=args.stu_name)
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # preprocess data
    train_path = f"./THUCNews/data/attack.txt"
    dev_path = f"./THUCNews/data/dev.txt"
    test_path = f"./THUCNews/data/test.txt"
    train_data = utils.MyDataSet(train_path, tokenizer)
    dev_data = utils.MyDataSet(dev_path, tokenizer)
    test_data = utils.MyDataSet(test_path, tokenizer)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    tea_model = Model()
    tea_model.load_state_dict(
        torch.load(
            f"./THUCNews/saved_dict/trigger/{args.tea_name}.ckpt", map_location=device
        )
    )

    #
    for index in range(5):
        model = importlib.import_module(f"models.{args.stu_name}")
        stu_model = model.Model()
        fit(index, tea_model, stu_model, train_loader, dev_loader, test_loader)
