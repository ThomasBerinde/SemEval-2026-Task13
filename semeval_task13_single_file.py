"""
Single-file reference project for SemEval-2026 Task 13.

This implementation follows the repository README constraints:
- Uses ONLY official SemEval Task 13 data (local parquet files or the official Kaggle release).
- Uses a general-purpose/code-oriented pretrained model (CodeBERT), not a third-party detector.
- Provides one dedicated method per subtask:
  * solve_subtask_a
  * solve_subtask_b
  * solve_subtask_c
- Produces submission files with columns: id,label (label is numeric id).

Usage examples:
  python semeval_task13_single_file.py --task A --output_dir runs/task_a
  python semeval_task13_single_file.py --task B --max_train_samples 50000
  python semeval_task13_single_file.py --task C --predict_only --model_dir runs/task_c/best
"""
from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Official data downloader snippet requested by task statement.
def download_official_data_with_kagglehub() -> str:
    import kagglehub

    path = kagglehub.dataset_download("daniilor/semeval-2026-task13")
    print("Path to dataset files:", path)
    return path


@dataclass
class TaskConfig:
    task_name: str
    parquet_relpath: str
    label_to_id_relpath: str


TASK_CONFIGS = {
    "A": TaskConfig("A", "task_A/task_a_trial.parquet", "task_A/label_to_id.json"),
    "B": TaskConfig("B", "task_B/task_b_trial.parquet", "task_B/label_to_id.json"),
    "C": TaskConfig("C", "task_C/task_c_trial.parquet", "task_C/label_to_id.json"),
}


def load_label_space(label_to_id_path: Path) -> Dict[str, int]:
    with open(label_to_id_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_code(code: str) -> str:
    # Light normalization only; no external data and no handcrafted detector features.
    code = "" if code is None else str(code)
    return code.strip()


def compute_macro_f1(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


class SemEvalTask13SingleFile:
    """
    One-method-per-subtask solution wrapper.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 256,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.seed = seed

    # --------- Helper methods (allowed by requirements) ---------
    def _load_dataframe(self, data_root: Path, task: str) -> pd.DataFrame:
        cfg = TASK_CONFIGS[task]
        parquet_path = data_root / cfg.parquet_relpath
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing task file: {parquet_path}. "
                "Use local repo data or download official data with kagglehub."
            )
        df = pd.read_parquet(parquet_path)

        required = {"code", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {parquet_path}: {sorted(missing)}")

        df = df.copy()
        df["code"] = df["code"].map(preprocess_code)
        df["label"] = df["label"].astype(int)
        if "id" not in df.columns:
            df["id"] = np.arange(len(df), dtype=int)
        return df

    def _build_datasets(
        self,
        df: pd.DataFrame,
        max_train_samples: Optional[int] = None,
        val_size: float = 0.1,
    ) -> Tuple[Dataset, Dataset]:
        if max_train_samples is not None and max_train_samples > 0 and len(df) > max_train_samples:
            df = df.sample(n=max_train_samples, random_state=self.seed).reset_index(drop=True)

        train_df, val_df = train_test_split(
            df[["code", "label"]],
            test_size=val_size,
            random_state=self.seed,
            stratify=df["label"],
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def _tokenize(batch):
            return tokenizer(
                batch["code"],
                truncation=True,
                max_length=self.max_length,
            )

        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

        train_ds = train_ds.map(_tokenize, batched=True)
        val_ds = val_ds.map(_tokenize, batched=True)

        train_ds = train_ds.rename_column("label", "labels")
        val_ds = val_ds.rename_column("label", "labels")

        return train_ds, val_ds

    def _train_and_save(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        num_labels: int,
        output_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> Path:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
        )

        args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=1,
            report_to="none",
            seed=self.seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_macro_f1,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Validation metrics:", metrics)

        best_dir = output_dir / "best"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        return best_dir

    def _predict_to_submission(
        self,
        model_dir: Path,
        df: pd.DataFrame,
        submission_path: Path,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        ds = Dataset.from_pandas(df[["id", "code"]].reset_index(drop=True))

        def _tokenize(batch):
            return tokenizer(batch["code"], truncation=True, max_length=self.max_length)

        ds = ds.map(_tokenize, batched=True)

        args = TrainingArguments(
            output_dir=str(model_dir / "_pred_tmp"),
            per_device_eval_batch_size=32,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, tokenizer=tokenizer)
        pred_out = trainer.predict(ds.remove_columns(["id", "code"]))

        pred_labels = np.argmax(pred_out.predictions, axis=1).astype(int)
        submission_df = pd.DataFrame({"id": df["id"].astype(int), "label": pred_labels})
        submission_df.to_csv(submission_path, index=False)
        print(f"Saved submission file: {submission_path}")

    def _solve_task(
        self,
        task: str,
        data_root: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        max_train_samples: Optional[int],
        predict_only: bool,
        model_dir: Optional[str],
    ) -> Dict[str, str]:
        data_root_path = Path(data_root)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = self._load_dataframe(data_root_path, task)
        cfg = TASK_CONFIGS[task]
        label_to_id = load_label_space(data_root_path / cfg.label_to_id_relpath)

        if predict_only:
            if not model_dir:
                raise ValueError("--predict_only requires --model_dir")
            chosen_model_dir = Path(model_dir)
        else:
            train_ds, val_ds = self._build_datasets(df, max_train_samples=max_train_samples)
            num_labels = int(df["label"].nunique())
            if num_labels != len(label_to_id):
                print(
                    f"Warning: observed {num_labels} labels in parquet but {len(label_to_id)} in label mapping."
                )
            chosen_model_dir = self._train_and_save(
                train_ds=train_ds,
                val_ds=val_ds,
                num_labels=num_labels,
                output_dir=out_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        submission_path = out_dir / f"task_{task.lower()}_submission.csv"
        self._predict_to_submission(chosen_model_dir, df, submission_path)

        return {
            "task": task,
            "model_dir": str(chosen_model_dir),
            "submission": str(submission_path),
        }

    # --------- Required: one method per subtask ---------
    def solve_subtask_a(
        self,
        data_root: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_train_samples: Optional[int] = None,
        predict_only: bool = False,
        model_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        return self._solve_task(
            task="A",
            data_root=data_root,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_train_samples=max_train_samples,
            predict_only=predict_only,
            model_dir=model_dir,
        )

    def solve_subtask_b(
        self,
        data_root: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_train_samples: Optional[int] = None,
        predict_only: bool = False,
        model_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        return self._solve_task(
            task="B",
            data_root=data_root,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_train_samples=max_train_samples,
            predict_only=predict_only,
            model_dir=model_dir,
        )

    def solve_subtask_c(
        self,
        data_root: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_train_samples: Optional[int] = None,
        predict_only: bool = False,
        model_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        return self._solve_task(
            task="C",
            data_root=data_root,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_train_samples=max_train_samples,
            predict_only=predict_only,
            model_dir=model_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file SemEval-2026 Task 13 solution")
    parser.add_argument("--task", choices=["A", "B", "C"], required=True)
    parser.add_argument("--data_root", default=".", help="Path containing task_A/task_B/task_C folders")
    parser.add_argument("--output_dir", default="runs/default")
    parser.add_argument("--model_name", default="microsoft/codebert-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--download_with_kagglehub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.download_with_kagglehub:
        download_official_data_with_kagglehub()

    system = SemEvalTask13SingleFile(model_name=args.model_name, max_length=args.max_length)

    if args.task == "A":
        result = system.solve_subtask_a(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            predict_only=args.predict_only,
            model_dir=args.model_dir,
        )
    elif args.task == "B":
        result = system.solve_subtask_b(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            predict_only=args.predict_only,
            model_dir=args.model_dir,
        )
    else:
        result = system.solve_subtask_c(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            predict_only=args.predict_only,
            model_dir=args.model_dir,
        )

    print("Done:", result)


if __name__ == "__main__":
    main()