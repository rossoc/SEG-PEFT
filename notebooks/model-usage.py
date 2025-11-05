# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Monte_Carlo_Attention
#     language: python
#     name: python3
# ---

# %%

# %% [markdown]
# # LoRapid


# %%
import torch
from transformers import (
    TrainingArguments,
    Trainer,
)
from lorapid import kvasir_dataset, compute_metrics, segformer, set_seed
import time
import yaml
import pandas as pd
import os
import zipfile
from peft import get_peft_model, LoraConfig

set_seed(42)

# %% [markdown]
# ## Dataset
# You can check out the dataset at the following link
# [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).

# %%

dataset_dir = "../data/Kvasir-SEG"
os.makedirs(dataset_dir, exist_ok=True)
!wget https://datasets.simula.no/downloads/kvasir-seg.zip -O kvasir-seg.zip

with zipfile.ZipFile("kvasir-seg.zip", "r") as zip_ref:
    zip_ref.extractall(dataset_dir)


# %% [markdown]
# ## Train [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) FFT


# %%


def train_segformer_fft(epochs, lr, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_size = 0.2
    model, model_name, _ = segformer()
    train_dataset, test_dataset = kvasir_dataset(model_name, test_size)

    training_args = TrainingArguments(
        output_dir="./outputs/" + save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=0,
        eval_steps=len(train_dataset),
        logging_steps=int(len(train_dataset) / 5),
        learning_rate=lr,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to=None,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        logging_dir=f"./outputs/{save_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,  # type: ignore
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time() - start_time

    all_metrics = {
        "training_history": trainer.state.log_history,
        "final_evaluation": trainer.evaluate(),
        "training_time": end_time,
    }

    with open(f"./outputs/{save_dir}/all_metrics.json", "w") as f:
        yaml.dump(all_metrics, f, indent=2)

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(f"./outputs/{save_dir}/training_history.csv", index=False)
    trainer.save_model(f"./outputs/{save_dir}/final")
    return trainer


# %%
epochs = 30
learning_rate = 5e-5
save_dir = "test_transformer_fft"


# %%
fft_trainer = train_segformer_fft(epochs, learning_rate, save_dir)


# %% [markdown]
# ## Train
# [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) with
# LoRA.
# Namely, we use [PEFT](https://github.com/huggingface/peft) to implmenent LoRA.


# %%
def train_segformer_lora(epochs, lr, r, lora_alpha, lora_dropout, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_size = 0.2
    model, model_name, modules = segformer()

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    train_dataset, test_dataset = kvasir_dataset(model_name, test_size)

    training_args = TrainingArguments(
        output_dir="./outputs/" + save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=0,
        eval_steps=len(train_dataset),
        logging_steps=int(len(train_dataset) / 5),
        learning_rate=lr,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to=None,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        logging_dir=f"./outputs/{save_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,  # type: ignore
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time() - start_time

    all_metrics = {
        "training_history": trainer.state.log_history,
        "final_evaluation": trainer.evaluate(),
        "training_time": end_time,
    }

    with open(f"./outputs/{save_dir}/all_metrics.json", "w") as f:
        yaml.dump(all_metrics, f, indent=2)

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(f"./outputs/{save_dir}/training_history.csv", index=False)
    trainer.save_model(f"./outputs/{save_dir}/final")
    return trainer


# %%
epochs = 30
learning_rate = 5e-5
rank = 8
lora_alpha = 32
lora_dropout = 0.1
save_dir = "test_transformer_lora"

# %%
fft_trainer = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)
# %%
