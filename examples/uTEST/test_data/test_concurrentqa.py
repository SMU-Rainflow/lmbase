"""
Independent test for ConcurrentQA dataset loading and formatting.

Checks:
- Registry load for `train` split
- Standardized sample content (question, groundtruth)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    # Run the test
    python examples/uTEST/test_data/test_concurrentqa.py

Expected Output:
    - Dataset: <lmbase.dataset.concurrentqa.ConcurrentQADataset object>
    - Standardized sample: VisualTextSample with question, groundtruth, cot_answer fields
    - Question: The formatted question with document context
    - Groundtruth: The expected answer string
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - ConcurrentQA dataset will be loaded from HuggingFace (stanfordnlp/concurrentqa-retrieval)
    - Internet connection required for initial dataset download
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():

    ds = dataset_registry.get(
        {
            "data_name": "concurrentqa",
            "data_path": "EXPERIMENT/data/concurrentqa",
        },
        "train",
    )

    print("Dataset:", ds)

    sample = ds[0]

    print("\nStandardized sample:")
    print(sample)

    print("\nQuestion:")
    print(sample["question"])

    print("\nGroundtruth:")
    print(sample["groundtruth"])

    msg = formatter.map_sample(sample, to_format="message")

    print("\nMessage format:")
    print(msg)

    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")

    print("\nFormatted via dataset hook:")
    print(ds[0])


if __name__ == "__main__":
    run()
