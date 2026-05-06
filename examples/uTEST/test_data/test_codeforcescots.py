"""
Independent test for CodeForces-CoTS dataset loading and formatting.

Checks:
- Registry load for `train` split with default subset (solutions_py)
- Standardized sample content (question, cot_answer, groundtruth, test_cases)
- generation field parsing: 上升趋势... 与此同时 -> cot_answer, code -> groundtruth
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    # Run the test
    python examples/uTEST/test_data/test_codeforcescots.py

Expected Output:
    - Dataset: <lmbase.dataset.codeforcescots.CodeForcesCoTSDataset object>
    - Standardized sample: TextCodeSample with question, cot_answer, groundtruth, test_cases
    - Question: The formatted competitive programming prompt
    - Cot answer: The chain-of-thought reasoning (from 上升趋势... 与此同时)
    - Groundtruth: The code solution (from after 与此同时)
    - Test cases: List of input/output example pairs
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - CodeForces-CoTS dataset will be loaded from HuggingFace (open-r1/codeforces-cots)
    - Internet connection required for initial dataset download
    - Default subset: solutions_py (9,556 examples)
    - Other subsets: solutions, checker_interactor, solutions_decontaminated,
      solutions_py_decontaminated
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load CodeForces-CoTS `train` split (solutions_py) and verify standardized
    and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "codeforcescots",
            "data_path": "EXPERIMENT/data/codeforcescots",
            "subset": "solutions_py",
        },
        "train",
    )

    print("Dataset:", ds)

    # Read the first sample
    s = ds[0]

    print("Standardized sample:")
    print(s)

    print("\nQuestion (prompt):")
    print(s["question"][:500] + "..." if len(s["question"]) > 500 else s["question"])

    print("\nCot answer (reasoning from 上升趋势...):")
    print(
        s["cot_answer"][:500] + "..." if len(s["cot_answer"]) > 500 else s["cot_answer"]
    )

    print("\nGroundtruth (code after 与此同时):")
    print(
        s["groundtruth"][:500] + "..."
        if len(s["groundtruth"]) > 500
        else s["groundtruth"]
    )

    print("\nTest cases:")
    test_cases = s.get("test_cases")
    if test_cases:
        for i, tc in enumerate(test_cases[:3]):
            print(f"  Example {i+1}:")
            print(f"    Input:  {str(tc['input'])[:100]}")
            print(f"    Output: {str(tc['output'])[:100]}")
    else:
        print("  (none)")

    print("\nSample info:")
    for k, v in s["sample_info"].items():
        print(f"  {k}: {v}")

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
