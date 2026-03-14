"""
Independent test for FinDER dataset loading and formatting.

Checks:
- Registry load for `train` split
- Standardized sample content (question, groundtruth)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    # Run the test
    python examples/uTEST/test_data/test_finder.py

Expected Output:
    - Dataset: <lmbase.dataset.finder.FinDERDataset object>
    - Standardized sample: TextSample with question, groundtruth, cot_answer fields
    - Question: The formatted question with evidence context
    - Groundtruth: The expected answer string
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - FinDER dataset will be loaded from HuggingFace (Linq-AI-Research/FinDER)
    - Internet connection required for initial dataset download
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load FinDER `train` split and verify standardized and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "finder",
            "data_path": "EXPERIMENT/data/finder",
        },
        "train",
    )

    print("Dataset:", ds)

    # 读取第一个样本
    s = ds[0]

    print("Standardized sample:")
    print(s)

    print("\nQuestion:")
    print(s["question"][:1000] + "..." if len(s["question"]) > 1000 else s["question"])

    print("\nGroundtruth:")
    print(s["groundtruth"])

    # 转换为 message 格式
    f = formatter.map_sample(s, to_format="message")

    print("\nMessage format:")
    print(f)

    # dataset hook
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")

    print("\nFormatted via dataset hook:")
    print(ds[0])


if __name__ == "__main__":
    run()
