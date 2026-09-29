import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AlbertForSequenceClassification,
    AlbertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    ReformerForSequenceClassification,
    ReformerTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import set_seed

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
set_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = "data/hmc/"
modelName = ""
pretrained_model_path = "models/"
MAX_SEQ_LEN = 64
EPOCHS = 4
lr = 1e-5
batches = [16, 24, 32]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", help="Name of the model to be used")
args = parser.parse_args()

if args.model:
    modelName = args.model

print("Model Name is:", modelName)


def Get_Model(modelName):
    if modelName == "XLNet":
        model = XLNetForSequenceClassification.from_pretrained(
            "xlnet-large-cased",
            num_labels=2,
        )
    elif modelName == "BERT":
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_path + "bert-large/",
            num_labels=2,
        )
    elif modelName == "RoBerta":
        model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_path + "roberta-large",
            num_labels=2,
        )
    elif modelName == "Albert":
        model = AlbertForSequenceClassification.from_pretrained(
            pretrained_model_path + "albert-xxlarge-v2/",
            num_labels=2,
        )
    elif modelName == "bertweet":
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_path + "bertweet/",
            num_labels=2,
        )
    elif modelName == "reformer":
        model = ReformerForSequenceClassification.from_pretrained(
            "google/reformer-crime-and-punishment",
            num_labels=2,
        )
    elif modelName == "electra":
        model = ElectraForSequenceClassification.from_pretrained(
            pretrained_model_path + "electra/",
            num_labels=2,
        )
    elif modelName == "deberta":
        model = DebertaForSequenceClassification.from_pretrained(
            pretrained_model_path + "deberta-large/",
            num_labels=2,
        )
    elif modelName == "gpt":
        model_config = GPT2Config.from_pretrained(
            pretrained_model_path + "gpt-2-large/",
            num_labels=2,
        )
        model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_path + "gpt-2-large/",
            config=model_config,
        )
        model.config.pad_token_id = model.config.eos_token_id

    return model


def Get_Encodings(modelName, inputText):
    input_ids = []
    attention_masks = []

    if modelName == "XLNet":
        tokenizer = XLNetTokenizer.from_pretrained(
            "xlnet-large-cased",
            do_lower_case=True,
        )
    elif modelName == "BERT":
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_path + "bert-large/",
            do_lower_case=True,
        )
    elif modelName == "RoBerta":
        tokenizer = RobertaTokenizer.from_pretrained(
            pretrained_model_path + "roberta-large/",
            do_lower_case=True,
        )
    elif modelName == "Albert":
        tokenizer = AlbertTokenizer.from_pretrained(
            pretrained_model_path + "albert-xxlarge-v2/",
            do_lower_case=True,
        )
    elif modelName == "bertweet":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path + "bertweet/",
            do_lower_case=True,
        )
    elif modelName == "reformer":
        tokenizer = ReformerTokenizer.from_pretrained(
            "google/reformer-crime-and-punishment",
            do_lower_case=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif modelName == "electra":
        tokenizer = ElectraTokenizer.from_pretrained(
            pretrained_model_path + "electra/",
            do_lower_case=True,
        )
    elif modelName == "deberta":
        tokenizer = DebertaTokenizer.from_pretrained(
            pretrained_model_path + "deberta-large/",
            do_lower_case=True,
        )
    elif modelName == "gpt":
        tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_path + "gpt-2-large/",
            do_lower_case=True,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    for sent in inputText:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    return input_ids, attention_masks


def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for step, d in enumerate(data_loader):
        input_ids = d[0].type(torch.LongTensor).to(device)
        attention_mask = d[1].type(torch.FloatTensor).to(device)
        targets = d[2].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets,
        )
        loss, _ = outputs[:2]

        loss.backward()
        tr_loss += loss.item()

        if nb_tr_steps % 40 == 0:
            print(
                "Train batch loss: {}".format(loss.item()),
                "For step No:",
                nb_tr_steps,
            )

        nb_tr_steps += 1
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_model(model, data_loader, device):
    model = model.eval()
    val_loss = 0
    nb_val_steps = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d[0].to(device)
            attention_mask = d[1].to(device)
            targets = d[2].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets,
            )
            loss, _ = outputs[:2]

            val_loss += loss.item()
            nb_val_steps += 1

            if nb_val_steps % 40 == 0:
                print(
                    "Val batch loss: {}".format(loss.item()),
                    "For step No:",
                    nb_val_steps,
                )

    return val_loss / nb_val_steps


def get_predictions(model, data_loader):
    model = model.eval()
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d[0].to(device)
            attention_mask = d[1].to(device)
            targets = d[2].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets,
            )
            logits = outputs[1]
            _, preds = torch.max(logits, dim=1)
            probs = F.softmax(logits, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return predictions, prediction_probs, real_values


for batch_size in batches:
    df_train = pd.read_csv(
        data_path + "train.csv",
        sep=",",
        encoding="utf-8",
        names=["tweet_id", "texts", "disease", "labels"],
        usecols=range(4),
    )
    df_train = shuffle(df_train)

    df_valid = pd.read_csv(
        data_path + "valid.csv",
        sep=",",
        encoding="utf-8",
        names=["tweet_id", "texts", "disease", "labels"],
        usecols=range(4),
    )

    df_test = pd.read_csv(
        data_path + "valid.csv",
        sep=",",
        encoding="utf-8",
        names=["tweet_id", "texts", "disease", "labels"],
        usecols=range(4),
    )

    print("Train Df -lenght: ", len(df_train))
    print("Valid Df -lenght: ", len(df_valid))
    print("Test Df -lenght: ", len(df_test))

    train_sentences = df_train.texts.str.lower().to_list()
    train_labels = df_train.labels.to_list()

    valid_sentences = df_valid.texts.str.lower().to_list()
    valid_labels = df_valid.labels.to_list()

    test_sentences = df_test.texts.str.lower().to_list()
    test_labels = df_test.labels.to_list()

    input_ids, attention_masks = Get_Encodings(modelName, train_sentences)
    train_input_ids = torch.cat(input_ids, dim=0)
    train_attention_masks = torch.cat(attention_masks, dim=0)
    train_labels = torch.tensor(train_labels)

    valid_input_ids, valid_attention_masks = Get_Encodings(
        modelName,
        valid_sentences,
    )
    valid_input_ids = torch.cat(valid_input_ids, dim=0)
    valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
    valid_labels = torch.tensor(valid_labels)

    train_dataset = TensorDataset(
        train_input_ids,
        train_attention_masks,
        train_labels,
    )
    valid_dataset = TensorDataset(
        valid_input_ids,
        valid_attention_masks,
        valid_labels,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
    )
    validation_dataloader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=batch_size,
    )

    model = Get_Model(modelName)
    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    history = defaultdict(list)
    best_loss = np.inf

    for epoch in range(EPOCHS):
        print("Epoch No: ", str(epoch + 1))

        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            scheduler,
        )
        print("Train loss:", str(train_loss))

        val_loss = eval_model(
            model,
            validation_dataloader,
            device,
        )
        print("Val loss:", str(val_loss))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            torch.save(
                model.state_dict(),
                "fine_tuned/" + modelName + "_batch_size_" + str(batch_size) + ".bin",
            )
            best_loss = val_loss

        print("best_loss: ", str(best_loss))

    if not os.path.exists("training_history/" + modelName):
        os.makedirs("training_history/" + modelName)

    pd.DataFrame(history).to_csv(
        "training_history/" + modelName + "_batch_size_" + str(batch_size) + ".csv"
    )

