"""
Interface of the FinCoT dataset.

Dataset Source: https://huggingface.co/datasets/TheFinAI/FinCoT

Description:
    A financial reasoning dataset with GPT-4o-generated chain-of-thought reasoning
    paths, derived from FinQA, ConvFinQA, TATQA, DocMath-Eval, Econ-Logic,
    BizBench-QA, and DocFinQA. Designed to enhance structured financial question
    answering with both positive and negative reasoning examples.

Size: ~184 MB (SFT: 7,686 examples, RL: 1,500 examples)

Splits:
    - SFT: Supervised fine-tuning split (7,686 examples)
      Use `split="SFT"` in config (fincot.py 42-43, 49)
    - RL: Reinforcement learning split (1,500 examples)
      Use `split="RL"` in config (fincot.py 42-43, 49)

Features:
    - Question: The financial question text
    - Reasoning_process: GPT-4o-generated chain-of-thought reasoning (positive)
    - Final_response: The final answer to the question
    - Negative_reasoning_process: Incorrect reasoning path (for RL contrastive training)
    - Negative_response: Incorrect final answer (for RL contrastive training)

License: Requires manual verification — see dataset repository for details.
Paper: arXiv:2502.08127 (Fin-o1)
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class FinCoTDataset(VisualTextBase):
    """A consistent interface for the FinCoT dataset."""

    def to_format(self, sample):
        """Convert raw sample to standardized format.

        Field mapping (fincot.py 62-81):
            Question                  -> question
            Reasoning_process         -> cot_answer
            Final_response            -> groundtruth
            Negative_reasoning_process -> sample_info.negative_reasoning
            Negative_response         -> sample_info.negative_response
        """
        self.idx += 1

        # Extract core fields
        question = sample["Question"].strip()
        reasoning = sample["Reasoning_process"] or ""
        final_response = sample["Final_response"] or ""

        # Append solution format prompt to question
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # Use Final_response as groundtruth
        groundtruth = final_response.strip()

        # Use Reasoning_process as chain-of-thought answer
        cot_answer = reasoning.strip()

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "negative_reasoning": sample["Negative_reasoning_process"],
                "negative_response": sample["Negative_response"],
            },
        )
