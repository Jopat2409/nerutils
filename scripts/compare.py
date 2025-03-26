import sys
import json

from collections import defaultdict

import matplotlib.pyplot as plt

def create_graph(infile: str, entity_level: bool) -> dict:

    print("Creaging frap")
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    errors = defaultdict(int)

    for example in filter(lambda x: x["incorrect"], data.get("errors", [])):

        entity_spans = []
        pointer = 0
        while pointer < len(example["expected"]):
            if example["expected"][pointer].startswith("B-"):
                try:
                    ent_end = next(i for i, tag in enumerate(example["expected"][pointer+1::]) if tag.startswith("B-") or tag=="O") + pointer + 1
                except StopIteration:
                    entity_spans.append(list(range(pointer, len(example["expected"]))))
                    break

                entity_spans.append(list(range(pointer, ent_end)))
                pointer = ent_end
                print(example)
                print(entity_spans)
            else:
                pointer +=1

        for ent in entity_spans:
            ent_type = example["expected"][ent[0]][2::]
            if not all(example["predicted"][i][2::] == ent_type for i in ent):
                errors[(ent_type, )]

        for incorrect in example["incorrect"]:
            errors[(example["expected"][incorrect], example["predicted"][incorrect])] += 1

    plt.barh([f"{e[0]} -> {e[1]}" for e in errors.keys()], errors.values())
    plt.show()

    return errors


if __name__ == "__main__":

    if1, if2 = sys.argv[1:3]

    entity_level = "--entity-level" in sys.argv
    ignore_iob = "--ignore-iob" in sys.argv


    err_1 = create_graph(if1, entity_level)
    err_2 = create_graph(if2, entity_level)

    plt.barh([f"{e[0]} -> {e[1]}" for e in err_1.keys()], [err_2[k] - err_1[k] for k in err_1])
    plt.show()

