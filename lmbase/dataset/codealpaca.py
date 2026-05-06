"""
Interface of the CodeAlpaca dataset.

Dataset Source: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

Description:
    A collection of ~20,000 code instruction-following examples generated using
    the Stanford Alpaca approach. Contains programming tasks with instructions
    and corresponding code solutions.

Size: ~20,000 rows

Splits:
    - train: ~20,000 examples (single split)

Features:
    - instruction: Natural language instruction
    - input: Optional input context
    - output: Code solution

License: CC BY 4.0
Language: English (instructions), Multiple programming languages (code)
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class CodeAlpacaDataset(VisualTextBase):
    """A consistent interface for the CodeAlpaca dataset."""

    def to_format(self, sample):
        """Convert raw sample to standardized format.

        Field mapping (codealpaca.py 31-48):
            instruction -> question (task description)
            input       -> question (appended as context, if present)
            output      -> groundtruth
            (no CoT)    -> cot_answer = ""
        """
        self.idx += 1

        # Build question: instruction + optional input context
        question = sample["instruction"]
        input_text = sample.get("input") or ""
        if input_text:
            question = f"{question}\nInput: {input_text}"

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=sample["output"],
            sample_info={
                "dataset": self.hf_dataname,
                "input": input_text,
            },
        )
