"""
Interface of the HotpotQA dataset.

The HotpotQA dataset is a question answering dataset that requires multi-hop reasoning.
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class HotpotQADataset(VisualTextBase):
    """A consistent interface for the HotpotQA dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Extract question and answer
        question = sample["question"]
        answer = sample.get("answer", "")

        # Build the question with context if available
        context = sample.get("context", "")
        if context:
            question = f"Context: {context}\n\nQuestion: {question}"

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
                "supporting_docs": sample.get("supporting_docs", []),
                "level": sample.get("level", ""),
            },
        )