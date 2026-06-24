from pathlib import Path

from EAP_forComponent.data import EAPComponentCollator, EAPComponentPairDataset
from EAP_forComponent.data import default_data_path as component_default_data_path


def default_data_path(dataset_name: str) -> Path:
	root = Path(__file__).resolve().parent
	candidate = root / "data" / f"{dataset_name}.csv"
	if candidate.exists():
		return candidate
	return component_default_data_path(dataset_name)


def load_pair_dataset(
	dataset_name: str,
	tokenizer,
	data_path: str | Path | None = None,
	corruption_column: str = "corrupted",
	max_samples: int | None = None,
	max_length: int | None = None,
	input_format: str = "auto",
) -> tuple[EAPComponentPairDataset, EAPComponentCollator]:
	csv_path = Path(data_path) if data_path is not None else default_data_path(dataset_name)
	dataset = EAPComponentPairDataset(
		csv_path=csv_path,
		dataset_name=dataset_name,
		corruption_column=corruption_column,
		max_samples=max_samples,
	)
	collator = EAPComponentCollator(
		tokenizer=tokenizer,
		dataset_name=dataset_name,
		max_length=max_length,
		input_format=input_format,
	)
	return dataset, collator


__all__ = ["default_data_path", "load_pair_dataset"]
