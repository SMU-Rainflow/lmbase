"""
Interface of the ConcurrentQA dataset.
"""
from datasets import load_dataset
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class ConcurrentQADataset(VisualTextBase):

    def __init__(self, split="train", hf_dataname=None, config=None):
        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        self.hf_dataset = load_dataset(
            self.hf_dataname,
            split=self.split
        )

    def to_format(self, sample):

        question = sample["question"]
        answer = sample["answers"][0]

        pos_docs = sample["pos_paras"]
        neg_docs = sample["neg_paras"]

        docs = pos_docs + neg_docs

        context = ""

        for i, d in enumerate(docs):
            context += f"\nDocument {i+1}: {d['title']}\n{d['text']}\n"

        question_with_context = f"""
Answer the question using the following documents.

Context:
{context}

Question:
{question}
"""

        return VisualTextSample(
            question=question_with_context,
            groundtruth=answer,
            cot_answer=answer,
            main_id=sample["_id"],
            split=self.split,
            sample_info=sample
        )