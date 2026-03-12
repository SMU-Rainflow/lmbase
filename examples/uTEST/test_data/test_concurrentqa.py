"""
Independent test for ConcurrentQA dataset loading and formatting.
"""
from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():

    ds = dataset_registry.get(
        {
            "data_name": "concurrentqa",
            "hf_dataname": "stanfordnlp/concurrentqa-retrieval",
            "data_path": "EXPERIMENT/data/concurrentqa",
        },
        "train",
    )

    print("Dataset:", ds)

    sample = ds[0]

    print("\nStandardized sample:")
    print(sample)

    print("\nQuestion:")
    print(sample['question'])

    print("\nGroundtruth:")
    print(sample['groundtruth'])

    msg = formatter.map_sample(sample, to_format="message")

    print("\nMessage format:")
    print(msg)

    ds.lm_format_function = lambda x: formatter.map_sample(
        x, to_format="message"
    )

    print("\nFormatted via dataset hook:")
    print(ds[0])


if __name__ == "__main__":
    run()