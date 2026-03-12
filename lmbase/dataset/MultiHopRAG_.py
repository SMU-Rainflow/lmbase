"""
Interface of the MultiHopRAG dataset.
"""

import os
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class MultiHopRAGDataset(VisualTextBase):
    """Interface for the MultiHopRAG dataset."""

    def __init__(
        self,
        split: str = "train",
        hf_dataname: str = "yixuantt/MultiHopRAG",
        config: dict = None,
    ):
        self.data_path = config.get("data_path", "./data")

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Convert HF sample to VisualTextSample format."""

        sample_id = self.idx
        self.idx += 1

        # question
        question = sample["query"].strip()

        # evidence documents
        evidence_list = sample.get("evidence_list", [])

        documents = []
        for i, ev in enumerate(evidence_list):
            title = ev.get("title", "")
            fact = ev.get("fact", "")
            source = ev.get("source", "")

            doc_text = f"Document {i+1} ({source}) - {title}: {fact}"
            documents.append(doc_text)

        context = "\n".join(documents)

        full_question = (
            f"Question:\n{question}\n\n"
            f"Documents:\n{context}\n\n"
            "Please answer the question based on the documents."
        )

        full_question = f"{full_question}{self.SOLUTION_FORMAT_PROMPT}"

        cot_answer = ""

        groundtruth = sample["answer"].strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=full_question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=[],
            sample_info={
                "dataset": self.hf_dataname,
                "question_type": sample.get("question_type", ""),
            },
        )