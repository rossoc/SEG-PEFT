#!/usr/bin/env python
"""
Training script for SegFormer model on Kvasir-SEG dataset.
This script trains a segmentation model to identify polyps in medical images.
"""

import torch
from transformers import (
    TrainingArguments,
    Trainer,
)
from argparse import ArgumentParser
from lorapid import kvasir_dataset, compute_metrics, segformer, set_seed
import time
import yaml
import pandas as pd


def main(epochs, lr, save_dir):
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
        save_steps=len(train_dataset),
        eval_steps=len(train_dataset),
        logging_steps=int(len(train_dataset) / 5),
        learning_rate=lr,
        save_total_limit=2,
        prediction_loss_only=False,
        remove_unused_columns=True,
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
        "final_evaluation": trainer.evaluate(),
        "training_time": end_time,
        "training_history": trainer.state.log_history,
    }

    with open(f"./outputs/{save_dir}/all_metrics.json", "w") as f:
        yaml.dump(all_metrics, f, indent=2)

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(f"./outputs/{save_dir}/training_history.csv", index=False)
    trainer.save_model(f"./outputs/{save_dir}/final")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--epochs", default=30)
    ap.add_argument("--lr", default=5e-5)
    ap.add_argument("--save-dir")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    main(args.epochs, args.lr, args.save_dir)
