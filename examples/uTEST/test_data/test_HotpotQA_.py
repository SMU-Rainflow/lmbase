"""
Independent test for HotpotQA dataset loading and formatting.

Checks:
- Registry load for `train` split
- Standardized sample content (question, groundtruth)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load HotpotQA `train` split and verify standardized and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "hotpotqa",
            "hf_dataname": "hotpotqa/hotpot_qa",
            "hf_subset": "distractor",
            "data_path": "EXPERIMENT/data/hotpotqa",
        },
        "train",
    )

    print("Dataset:", ds)

    # 读取第一个样本
    s = ds[0]

    print("Standardized sample:")
    print(s)

    print("\nQuestion:")
    print(s['question'])

    print("\nGroundtruth:")
    print(s['groundtruth'])

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
