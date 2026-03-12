"""
Independent test for MultiHopRAG dataset loading and formatting.
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
            "hf_dataname": "yixuantt/MultiHopRAG",
            "data_path": "EXPERIMENT/data/multihoprag",
           "hf_subset": "MultiHopRAG",  # 指定配置
        },
        "train",
    )

    print("Dataset:", ds)

    # 获取一个 sample
    s = ds[0]

    print("\nStandardized sample:")
    print(s)

    print("\nQuestion:")
    print(s['question'])

    print("\nGroundtruth:")
    print(s['groundtruth'])

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