import sys
import json
import unicodedata
from datasets import load_dataset, ClassLabel

ontonotes_label = ClassLabel(num_classes=37, names=["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE",])

def get_entities(word_indices, tags):

    entities = []
    tag_names = [ontonotes_label.int2str(i) for i in tags]

    current_type, current_words = "O", []

    for i, tag in enumerate(tag_names):

        if current_type != 'O' and (tag == 'O' or tag.startswith('B')):
            entities.append({"start": word_indices[current_words[0]][0], "end": word_indices[current_words[-1]][1], "label": current_type[2:]})
            current_words = []
            current_type = tag

        if tag.startswith('B'):
            current_type = tag
            current_words = []

        current_words.append(i)

    return entities



def sentence_to_jsonl(words: list, tags: list) -> dict:
    words[-1].replace('/', '')
    text = ' '.join(words).replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']').replace('-RCB-', '}').replace('-LCB-', '{')
    if text.startswith("For instance , because the first character"):
        print(words)
        print(tags)
        print([unicodedata.normalize("NFKC", token) for token in words])
    word_indices = list(zip([0] + [i + 1 for i, ch in enumerate(text) if ch == " "], [i for i, ch in enumerate(text) if ch == " "] + [len(text)]))
    data = {"word_positions": word_indices, "entities": get_entities(word_indices, tags), "text": text}
    return data


def doc_to_jsonl(document: list, id_: str):
    data = []
    for i, sentence in enumerate(document):
        data.append({**sentence_to_jsonl(sentence["words"], sentence["named_entities"]), "id": f"{id_}-{i}"})

    return data



def convert(split: str):
    ontonotes = load_dataset("ontonotes/conll2012_ontonotesv5", "english_v12", trust_remote_code=True, split=split)
    results = []
    num_documents = len(ontonotes["sentences"])
    for i, sentences in enumerate(ontonotes["sentences"]):
        print(f"Converting document {i} / {num_documents}", end='\r')
        results.append({"id": ontonotes["document_id"][i], "examples": doc_to_jsonl(sentences, ontonotes["document_id"][i])})

    with open(f"ontonotes.{split}.jsonl", "w+", encoding='utf-8') as f:
        for document in results:
            f.write(f"{json.dumps(document, ensure_ascii=False)}\n")

if __name__ == "__main__":
    convert(sys.argv[1])
