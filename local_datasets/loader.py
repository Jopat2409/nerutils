import os

from typing import Generator, Dict
from itertools import chain

from datasets import Dataset, ClassLabel, load_dataset, Sequence, DatasetDict

def conll_huggingface_generator(path: str) -> Generator[Dict[str, list], None, None]:
    with open(path, "r", encoding='utf-8') as f:
        document = 0
        tags, tokens = [], []
        for line in [ln.strip() for ln in f.readlines()]:
            if (not line) and (tags and tokens):
                yield {"document": document, "tokens": tokens, "ner_tags":tags}
                tags, tokens = [], []
            elif line.startswith("-DOCSTART-"):
                document += 1
            elif line:
                token, *_, tag = line.split()
                tags.append(tag)
                tokens.append(token)
        if tags and tokens:
            yield {"document": document, "tokens": tokens, "ner_tags":tags}

def load_to_huggingface(path: str) -> Dataset:
    """Load a CoNLL-2003

    Args:
        path (str): _description_

    Returns:
        Dataset: _description_
    """

    if os.path.isdir(path):
        return DatasetDict({f.split(".")[0]: load_to_huggingface(os.path.join(path, f)) for f in os.listdir(path)})

    dataset = Dataset.from_generator(lambda: conll_huggingface_generator(path), split="test")

    class_labels = Sequence(ClassLabel(names=list(sorted(set(chain.from_iterable(dataset["ner_tags"]))))))
    dataset = dataset.cast_column("ner_tags", class_labels)

    return dataset


if __name__ == "__main__":
    print(load_dataset("conll2003", split="test", trust_remote_code=True).features["ner_tags"])
    print(load_to_huggingface("cleanconll/conll_03/test.txt").features["ner_tags"])
