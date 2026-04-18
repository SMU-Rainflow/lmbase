"""
Independent test for GAIA dataset loading and formatting.

Checks:
- Registry load for `validation` split
- Standardized sample content (question, groundtruth, level)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    python examples/utest/test_data/test_gaia.py

Expected Output:
    - Dataset: <lmbase.dataset.gaia.GAIADataset object>
    - Standardized sample: TextSample with question, groundtruth, cot_answer fields
    - Question: The formatted task question
    - Groundtruth: The expected final answer string
    - Level: Difficulty level (1, 2, or 3)
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - GAIA dataset will be loaded from HuggingFace (gaia-benchmark/GAIA)
    - Internet connection required for initial dataset download
    - HuggingFace login may be required (gated dataset)
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load GAIA `validation` split and verify standardized and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "gaia",
            "data_path": "EXPERIMENT/data/gaia",
            "subset": "2023_all",
        },
        "validation",
    )

    print("Dataset:", ds)

    # 读取第一个样本
    s = ds[0]

    print("\nStandardized sample:")
    print(s)

    print("\nQuestion:")
    print(s["question"])

    print("\nGroundtruth:")
    print(s["groundtruth"])

    print("\nLevel:")
    print(s["sample_info"]["level"])

    print("\nFile name:")
    print(s["sample_info"]["file_name"])

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
