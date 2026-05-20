import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import pandas as pd
import yaml

from rlie.datasets.tasks import BaseTask
from rlie.datasets.labels import extract_label_register


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TASK_SPEC_ROOT = PACKAGE_ROOT / "task_specs"
REPO_ROOT = PACKAGE_ROOT.parent


def _normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TaskDataLoader:
    """Wrapper around BaseTask to expose full datasets based on config."""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger("rlie.datasets.data_loader")

        task_config_path = self._resolve_task_config_path(config)
        config.setdefault("task_config", task_config_path)

        self.task_dir = Path(task_config_path).resolve().parent
        self._train_samples: Optional[int] = self._resolve_sample_count(
            config,
            primary_key="num_train_samples",
            legacy_key="max_train_samples",
        )
        self._val_samples: Optional[int] = self._resolve_sample_count(
            config,
            primary_key="num_valid_samples",
            legacy_key="max_valid_samples",
        )
        self._test_samples: Optional[int] = self._resolve_sample_count(
            config,
            primary_key="num_test_samples",
            legacy_key="max_test_samples",
        )
        self._data_seed: int = int(config.get("data_seed", config.get("dataset_seed", 42)))
        self.task = BaseTask(
            config_path=task_config_path,
            from_register=extract_label_register,
            use_ood=False,
        )
        self.label_values: List[str] = []
        self.label_mapping = self._determine_label_mapping(config)
        self.positive_label_text: Optional[str] = None
        self.negative_label_text: Optional[str] = None
        if self.label_values:
            if len(self.label_values) >= 1:
                self.positive_label_text = str(self.label_values[0])
            if len(self.label_values) >= 2:
                self.negative_label_text = str(self.label_values[1])

    def _resolve_task_config_path(self, config: Dict) -> str:
        if "task_config" in config:
            return config["task_config"]

        data_root = config.get("data_root")
        if isinstance(data_root, list):
            data_root = data_root[0] if data_root else None
        if not data_root:
            raise ValueError("Config must specify either 'task_config' or 'data_root'.")

        root_path = Path(data_root)
        candidates = []

        if root_path.suffix in {".yaml", ".yml"}:
            candidates.append(root_path)
        else:
            candidates.append(root_path / "task.yaml")
            candidates.append(root_path / "config.yaml")
            candidates.append(TASK_SPEC_ROOT / root_path / "task.yaml")
            candidates.append(TASK_SPEC_ROOT / root_path / "config.yaml")
            candidates.append(REPO_ROOT / "data" / root_path / "task.yaml")
            candidates.append(REPO_ROOT / "data" / root_path / "config.yaml")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not locate task config for data_root '{data_root}'. Checked: {rendered}"
        )

    def _determine_label_mapping(self, config: Dict) -> Dict[str, int]:
        explicit_mapping = config.get("label_mapping")
        if explicit_mapping:
            normalized_mapping = {}
            label_values: List[str] = []
            for label, idx in explicit_mapping.items():
                key = _normalize_label(label)
                if key in normalized_mapping:
                    raise ValueError(
                        f"Duplicate normalized label '{key}' detected in explicit mapping."
                    )
                normalized_mapping[key] = int(idx)
                label_values.append(str(label))
            # Preserve ordering only if explicitly provided; otherwise derive later
            if "label_values" not in config and label_values:
                config["label_values"] = label_values
                self.label_values = label_values
            else:
                self.label_values = config.get("label_values", [])
            return normalized_mapping

        metadata_path = self.task_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Unable to infer label mapping: metadata.json not found and no label_mapping provided."
            )

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        labels_section = metadata.get("labels", {})
        label_name = self.task.label_name
        if label_name not in labels_section:
            raise KeyError(
                f"Label column '{label_name}' not present in metadata.json; cannot infer mapping."
            )

        values = labels_section[label_name].get("values")
        if not values:
            raise ValueError(
                f"Metadata for label '{label_name}' does not define any values; cannot infer mapping."
            )

        self.logger.info(
            "Inferred label mapping for '%s' from metadata.json: %s",
            label_name,
            values,
        )
        if len(values) < 2:
            raise ValueError(
                f"Metadata for label '{label_name}' must define at least two values."
            )

        positive_label = values[0]
        negative_label = values[1]
        mapping = {
            _normalize_label(positive_label): 1,
            _normalize_label(negative_label): 0,
        }

        for extra_value in values[2:]:
            key = _normalize_label(extra_value)
            if key in mapping:
                raise ValueError(
                    f"Duplicate normalized label '{key}' detected in metadata.json for label '{label_name}'."
                )
            mapping[key] = 0  # treat additional labels as negative by default

        config["label_mapping"] = mapping
        config["label_values"] = values
        self.label_values = values
        return mapping

    @staticmethod
    def _resolve_sample_count(
        config: Dict,
        primary_key: str,
        legacy_key: str,
    ) -> Optional[int]:
        if primary_key in config:
            value = config[primary_key]
            return int(value) if value is not None else None
        if legacy_key in config:
            value = config[legacy_key]
            return int(value) if value is not None else None
        return None

    def load_dataset(self) -> DatasetBundle:
        train_df, test_df, val_df = self.task.get_data(
            num_train=self._train_samples,
            num_test=self._test_samples,
            num_val=self._val_samples,
            seed=self._data_seed,
        )
        pos_label, neg_label = self._determine_display_labels(train_df, val_df, test_df)
        self.positive_label_text = pos_label
        self.negative_label_text = neg_label

        for df in (train_df, val_df, test_df):
            if df is None:
                continue
            df["pos_label"] = pos_label
            df["neg_label"] = neg_label

        return DatasetBundle(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )

    def _determine_display_labels(
        self, *dataframes: pd.DataFrame
    ) -> Tuple[str, str]:
        label_column = getattr(self.task, "label_name", "label")

        def _candidate(idx: int, fallback: Optional[str]) -> Optional[str]:
            for df in dataframes:
                if df is None or label_column not in df:
                    continue
                series = df[label_column].dropna().unique().tolist()
                for value in series:
                    normalized = _normalize_label(value)
                    mapped = self.label_mapping.get(normalized)
                    if mapped == idx:
                        return str(value)
            return fallback

        positive_indices = [i for i in set(self.label_mapping.values()) if i != 0]
        positive_idx = min(positive_indices) if positive_indices else 1
        negative_idx = 0 if 0 in self.label_mapping.values() else (
            max(set(self.label_mapping.values())) if self.label_mapping else 0
        )

        pos_fallback = self.positive_label_text or (
            self.label_values[0] if self.label_values else None
        )
        neg_fallback = self.negative_label_text or (
            self.label_values[1] if len(self.label_values) >= 2 else None
        )

        pos_label = _candidate(positive_idx, pos_fallback) or "positive"
        neg_label = _candidate(negative_idx, neg_fallback) or "negative"

        return pos_label, neg_label


def load_config(path: str) -> Dict:
    def expand_env(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, list):
            return [expand_env(item) for item in value]
        if isinstance(value, dict):
            return {key: expand_env(item) for key, item in value.items()}
        return value

    with open(path, "r") as f:
        return expand_env(yaml.safe_load(f))
