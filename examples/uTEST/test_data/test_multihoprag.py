"""
Independent test for MultiHopRAG dataset loading and formatting.

Checks:
- Registry load for `train` split
- Standardized sample content (question, groundtruth)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    # Run the test
    python examples/uTEST/test_data/test_multihoprag.py

Expected Output:
    - Dataset: <lmbase.dataset.multihoprag.MultiHopRAGDataset object>
    - Standardized sample: VisualTextSample with question, groundtruth, cot_answer fields
    - Question: The formatted question with evidence documents
    - Groundtruth: The expected answer string
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - MultiHopRAG dataset will be loaded from HuggingFace (yixuantt/MultiHopRAG)
    - Internet connection required for initial dataset download
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load MultiHopRAG dataset and verify standardized and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "multihoprag",
            "data_path": "EXPERIMENT/data/multihoprag",
        },
        "train",
    )

    print("Dataset:", ds)

    # 获取一个 sample
    s = ds[0]

    print("\nStandardized sample:")
    print(s)

    print("\nQuestion:")
    print(s["question"])

    print("\nGroundtruth:")
    print(s["groundtruth"])

    # 转换为 message format
    f = formatter.map_sample(s, to_format="message")

    print("\nMessage format:")
    print(f)

    # dataset hook
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")

    print("\nFormatted via dataset hook:")
    print(ds[0])


if __name__ == "__main__":
    run()
