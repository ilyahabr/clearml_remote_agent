"""
Simple BERT training script using PyTorch Lightning on IMDB dataset.
This is a self-contained example for sentiment classification.
Device-agnostic: automatically uses GPU if available, otherwise falls back to CPU.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
from dotenv import load_dotenv
from clearml import Task
from loguru import logger


def get_device_info():
    """
    Detect available compute devices and return device information.

    Returns:
        dict: Dictionary containing device information including:
            - cuda_available: bool
            - device_count: int
            - device_name: str
            - device_type: str (cuda or cpu)
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
    }

    return device_info


def log_device_info():
    """Log device information."""
    device_info = get_device_info()

    logger.info("Device Information")
    logger.info(f"CUDA Available: {device_info['cuda_available']}")
    if device_info["cuda_available"]:
        logger.info(f"GPU Count: {device_info['device_count']}")
        logger.info(f"GPU Name: {device_info['device_name']}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Device Type: {device_info['device_type'].upper()}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    return device_info


class IMDBDataset(Dataset):
    """PyTorch Dataset wrapper for IMDB with BERT tokenization."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class IMDBDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for IMDB dataset."""

    def __init__(self, batch_size=16, max_length=128, num_workers=4, subset_size=1000):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.subset_size = subset_size  # Use subset for faster training
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def prepare_data(self):
        # Download dataset
        load_dataset("imdb")

    def setup(self, stage=None):
        # Load dataset
        dataset = load_dataset("imdb")

        # Use subset for faster training (remove this line for full dataset)
        train_data = dataset["train"].shuffle(seed=42).select(range(self.subset_size))
        test_data = (
            dataset["test"].shuffle(seed=42).select(range(self.subset_size // 4))
        )

        # Create datasets
        self.train_dataset = IMDBDataset(
            texts=train_data["text"],
            labels=train_data["label"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        self.val_dataset = IMDBDataset(
            texts=test_data["text"],
            labels=test_data["label"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class BERTSentimentClassifier(pl.LightningModule):
    """PyTorch Lightning Module for BERT-based sentiment classification."""

    def __init__(self, learning_rate=2e-5, num_labels=2, clearml_logger=None):
        super().__init__()
        self.save_hyperparameters(ignore=["clearml_logger"])

        # Load pretrained BERT model
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )

        self.learning_rate = learning_rate
        self.clearml_logger = clearml_logger  # Add ClearML logger

        # Store predictions for metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Log metrics to PyTorch Lightning
        self.log("train_loss", loss, prog_bar=True)

        # Log to ClearML
        if self.clearml_logger:
            self.clearml_logger.report_scalar(
                title="Loss",
                series="train",
                value=loss.item(),
                iteration=self.global_step,
            )

        # Store for epoch end
        self.training_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": preds.detach(),
                "labels": batch["labels"].detach(),
            }
        )

        return loss

    def on_train_epoch_end(self):
        # Calculate epoch metrics
        all_preds = torch.cat([x["preds"] for x in self.training_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.training_step_outputs])

        accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())

        self.log("train_acc", accuracy, prog_bar=True)

        # Log to ClearML
        if self.clearml_logger:
            self.clearml_logger.report_scalar(
                title="Accuracy",
                series="train",
                value=accuracy,
                iteration=self.current_epoch,
            )

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)

        # Store for epoch end
        self.validation_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": preds.detach(),
                "labels": batch["labels"].detach(),
            }
        )

        return loss

    def on_validation_epoch_end(self):
        # Calculate epoch metrics
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

        accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())

        f1 = f1_score(
            all_labels.cpu().numpy(), all_preds.cpu().numpy(), average="binary"
        )

        self.log("val_acc", accuracy, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        # Log to ClearML
        if self.clearml_logger:
            # Calculate average validation loss
            avg_val_loss = (
                torch.stack([x["loss"] for x in self.validation_step_outputs])
                .mean()
                .item()
            )

            self.clearml_logger.report_scalar(
                title="Loss",
                series="validation",
                value=avg_val_loss,
                iteration=self.current_epoch,
            )
            self.clearml_logger.report_scalar(
                title="Accuracy",
                series="validation",
                value=accuracy,
                iteration=self.current_epoch,
            )
            self.clearml_logger.report_scalar(
                title="F1 Score",
                series="validation",
                value=f1,
                iteration=self.current_epoch,
            )

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BERT sentiment classifier on IMDB dataset (Device-agnostic: CPU/GPU)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to use for training (auto=automatic detection, cpu=force CPU, gpu/cuda=force GPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=3,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help="Number of samples to use for training (use full dataset with -1)",
    )

    return parser.parse_args()


def main():
    """Main training function with device-agnostic setup."""

    # Parse command line arguments
    args = parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Log device information
    device_info = log_device_info()

    # Setup ClearML credentials
    clearml_api_key = os.getenv("CLEARML_API_ACCESS_KEY")
    clearml_secret_key = os.getenv("CLEARML_API_SECRET_KEY")

    if not clearml_api_key or not clearml_secret_key:
        raise ValueError(
            "ClearML credentials not found. Please ensure CLEARML_API_ACCESS_KEY "
            "and CLEARML_API_SECRET_KEY are set in your .env file."
        )

    # Set ClearML credentials
    Task.set_credentials(
        api_host="https://api.clear.ml",
        web_host="https://app.clear.ml",
        files_host="https://files.clear.ml",
        key=clearml_api_key,
        secret=clearml_secret_key,
    )

    # Initialize ClearML Task for logging
    task = Task.init(
        project_name="BERT-Sentiment-Classification",
        task_name="IMDB-Training-BERT-Base",
        task_type=Task.TaskTypes.training,
    )

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Hyperparameters
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = 128
    SUBSET_SIZE = args.subset_size

    # Log hyperparameters and device info to ClearML
    task.connect(
        {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "subset_size": SUBSET_SIZE,
            "model_name": "bert-base-uncased",
            "device_arg": args.device,
            "cuda_available": device_info["cuda_available"],
            "device_name": device_info["device_name"],
            "device_count": device_info["device_count"],
        }
    )

    # Initialize data module
    data_module = IMDBDataModule(
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        subset_size=SUBSET_SIZE,
    )

    # Initialize model
    model = BERTSentimentClassifier(
        learning_rate=LEARNING_RATE,
        num_labels=2,
        clearml_logger=task.get_logger(),  # Pass ClearML logger
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="bert-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")

    # Determine accelerator based on user input and availability
    if args.device == "cpu":
        accelerator = "cpu"
        devices = 1
        logger.info("Training on CPU (forced by user)")
    elif args.device in ["gpu", "cuda"]:
        if device_info["cuda_available"]:
            accelerator = "gpu"
            devices = 1
            logger.info(f"Training on GPU: {device_info['device_name']}")
        else:
            logger.warning("GPU requested but not available. Falling back to CPU")
            accelerator = "cpu"
            devices = 1
    else:  # auto
        if device_info["cuda_available"]:
            accelerator = "gpu"
            devices = 1
            logger.info(
                f"Training on GPU (auto-detected): {device_info['device_name']}"
            )
        else:
            accelerator = "cpu"
            devices = 1
            logger.info("Training on CPU (auto-detected)")

    # Log the selected device to ClearML
    task.get_logger().report_text(
        f"Training Device: {accelerator.upper()}\n"
        f"Device Name: {device_info['device_name']}\n"
        f"PyTorch Version: {torch.__version__}"
    )

    # Initialize trainer with device-agnostic configuration
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True,
        log_every_n_steps=10,
    )

    # Train the model
    logger.info(f"Starting training for {MAX_EPOCHS} epochs on {accelerator.upper()}")
    trainer.fit(model, data_module)

    # Log completion
    logger.success(
        f"Training completed! Best model: {checkpoint_callback.best_model_path}"
    )

    # Log final metrics to ClearML
    task.get_logger().report_text(
        f"Training completed! Best model saved at: {checkpoint_callback.best_model_path}\n"
        f"Trained on: {accelerator.upper()}"
    )


if __name__ == "__main__":
    main()
