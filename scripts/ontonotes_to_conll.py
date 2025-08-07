import os
import argparse

from datasets import load_dataset, ClassLabel, Dataset

ontonotes_label = ClassLabel(num_classes=37, names=["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE",])

def convert(split: Dataset) -> str:
    """Convert the given OntoNotes split to a string

    Args:
        split (Dataset): The OntoNotes split

    Returns:
        str: The string to write to the file
    """

    data = ""
    num_documents = len(split["sentences"])
    for i, examples in enumerate(split["sentences"]):
        print(f"Converting document {i} / {num_documents}", end='\r')
        data += "-DOCSTART- O\n\n"
        for example in examples:
            for word, tag in zip(example["words"], example["named_entities"]):
                data += f"{word} {ontonotes_label.int2str(tag)}\n"
            data += "\n"
    return data

def save_ontonotes(directory: str) -> None:
    """Save the OntoNotes dataset to CoNLL-2003 format text files

    Args:
        directory (str): The directory to save the OntoNotes files to
    """
    ontonotes = load_dataset("ontonotes/conll2012_ontonotesv5", "english_v12", trust_remote_code=True)

    for split in ontonotes:
        with open(os.path.join(directory, f"{split}.txt"), "w+", encoding='utf-8') as f:
            f.write(convert(ontonotes[split]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts the OntoNotes 5.0 NER dataset to CoNLL format')
    parser.add_argument('output', help="The output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    save_ontonotes(args.output)

