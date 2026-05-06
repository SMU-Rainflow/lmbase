"""
Interface of the hfcodealpaca dataset.

Dataset Source: https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K

Description:
    HuggingFace's version of CodeAlpaca with ~20,000 code instruction-following
    examples. Split into train (18,000) and test (2,000) sets.

Size: ~20,000 rows

Splits:
    - train: ~18,000 examples
    - test: ~2,000 examples

Features:
    - instruction/prompt: Natural language instruction
    - output/completion: Code solution

License: Creative Commons
Language: English (instructions), Multiple programming languages (code)
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class CodeAlpacaDataset(VisualTextBase):
    """A consistent interface for the hfCodeAlpaca dataset."""

    def to_format(self, sample):
        """Convert raw sample to standardized format.

        Field mapping (hfcodealpaca.py 30-55):
            instruction/prompt -> question (task description)
            input              -> question (appended as context, if present)
            output/completion  -> groundtruth
            (no CoT)           -> cot_answer = ""
        """
        self.idx += 1

        # Handle both instruction/output and prompt/completion schemas
        if "instruction" in sample:
            question = sample["instruction"]
            groundtruth = sample.get("output", "")
            input_text = sample.get("input") or ""
        else:
            question = sample.get("prompt", "")
            groundtruth = sample.get("completion", "")
            input_text = sample.get("input") or ""

        # Append input context if present
        if input_text:
            question = f"{question}\nInput: {input_text}"

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "input": input_text,
            },
        )
