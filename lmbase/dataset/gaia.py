"""
Interface of the GAIA dataset.

Dataset Source: https://huggingface.co/datasets/gaia-benchmark/GAIA

Description:
    GAIA (General AI Assistants) benchmark for evaluating general-purpose AI
    agents on real-world tasks requiring tool use, multi-step reasoning, and
    factual retrieval. Tasks span web search, file parsing, and multi-modal
    reasoning.

Configurations (subsets):
    - 2023_all: Full 2023 evaluation set
    Config setting in code: subset="2023_all"

Splits:
    - validation: ~165 examples
    - test: examples without ground truth

Features:
    - task_id: Unique task identifier
    - Question: The task question
    - Level: Difficulty level (1, 2, or 3)
    - Final answer: Ground truth answer
    - file_name: Optional attached file name
    - file_path: Optional attached file path
    - Annotator Metadata: Dict with Steps and other annotations

License: CC BY 4.0
"""

from datasets import load_dataset

from lmbase.dataset.base import TextSample, VisualTextBase


class GAIADataset(VisualTextBase):
    """A consistent interface for the GAIA benchmark dataset."""

    def map_dataset(self):
        subset_name = self.config.get("subset", "2023_all")
        self.hf_dataset = load_dataset(self.hf_dataname, subset_name, split=self.split)
        super().map_dataset()

    def to_format(self, sample):
        self.idx += 1

        question = sample["Question"]
        groundtruth = sample.get("Final answer", "") or ""
        level = str(sample.get("Level", ""))
        task_id = sample.get("task_id", f"ID{self.idx}")
        file_name = sample.get("file_name", "") or ""
        file_path = sample.get("file_path", "") or ""
        annotator_metadata = sample.get("Annotator Metadata", {}) or {}

        formatted_question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=task_id,
            split=self.split,
            question=formatted_question,
            cot_answer=annotator_metadata.get("Steps", "") if isinstance(annotator_metadata, dict) else "",
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "level": level,
                "file_name": file_name,
                "file_path": file_path,
                "annotator_metadata": annotator_metadata,
            },
        )
