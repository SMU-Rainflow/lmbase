"""
Independent test for FinCoT dataset loading and formatting.

Checks:
- Registry load for `SFT` split
- Standardized sample content (question, groundtruth, cot_answer)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    # Run the test
    python examples/uTEST/test_data/test_fincot.py

Expected Output:
    - Dataset: <lmbase.dataset.fincot.FinCoTDataset object>
    - Standardized sample: TextSample with question, groundtruth, cot_answer fields
    - Question: The formatted financial question
    - Groundtruth: The final response answer string
    - Cot answer: The chain-of-thought reasoning process
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - FinCoT dataset will be loaded from HuggingFace (TheFinAI/FinCoT)
    - Internet connection required for initial dataset download
    - Available splits: "SFT" (7,686 examples), "RL" (1,500 examples)
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load FinCoT `SFT` split and verify standardized and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "fincot",
            "data_path": "EXPERIMENT/data/fincot",
        },
        "SFT",
    )

    print("Dataset:", ds)

    # Read the first sample
    s = ds[0]

    print("Standardized sample:")
    print(s)

    print("\nQuestion:")
    print(s["question"][:500] + "..." if len(s["question"]) > 500 else s["question"])

    print("\nGroundtruth:")
    print(s["groundtruth"])

    print("\nCot answer (reasoning process):")
    print(
        s["cot_answer"][:500] + "..." if len(s["cot_answer"]) > 500 else s["cot_answer"]
    )

    print("\nNegative reasoning (for RL):")
    neg_reasoning = s["sample_info"]["negative_reasoning"]
    if neg_reasoning:
        print(
            neg_reasoning[:300] + "..." if len(neg_reasoning) > 300 else neg_reasoning
        )
    else:
        print("(none)")

    # Convert to message format
    f = formatter.map_sample(s, to_format="message")

    print("\nMessage format:")
    print(f)

    # Dataset hook
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")

    print("\nFormatted via dataset hook:")
    print(ds[0])


if __name__ == "__main__":
    run()
