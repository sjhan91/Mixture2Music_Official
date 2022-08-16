import json
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from glob import glob
from utils.data import *
from utils.model import *
from utils.common_utils import *


# initialize model with GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load config file
with open("./config.json", "r") as f:
    config = json.load(f)

# data here!
file_list = glob(config["train_folder_path"] + "/*")

num_data = len(file_list)
num_train = int(num_data * 0.9)

train_file_list = file_list[:num_train]
val_file_list = file_list[num_train:]

print("The number of train: %d" % len(train_file_list))
print("The number of validation: %d" % len(val_file_list))

# dataloader
train_params = {
    "batch_size": config["batch_size"],
    "shuffle": True,
    "pin_memory": True,
    "num_workers": 4,
}

val_params = train_params.copy()
val_params["shuffle"] = False

train_set = DataLoader(DatasetSampler(train_file_list), **train_params)
val_set = DataLoader(DatasetSampler(val_file_list), **val_params)

# model here!
model = TransUNet_Lightning(
    config["in_ch"],
    config["out_ch"],
    config["num_layers"],
    config["d_model"],
    config["latent_dim"],
    config["time_emb_dim"],
    config["time_steps"],
    rate=config["rate"],
)

model_name = "TransUNet-Mixture-" + "{epoch}-{val_loss:.4f}"
checkpoint_callback = ModelCheckpoint(filename=model_name, dirpath="./model/", monitor="val_loss")

trainer = pl.Trainer(
    num_nodes=1,
    max_epochs=50,
    gpus=config["gpus"],
    strategy=config["strategy"],
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_set, val_set)

print("best model path :", checkpoint_callback.best_model_path)
print("final results :", checkpoint_callback.best_model_score)
