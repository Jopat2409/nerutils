"""
Functions used to evaluate and test the implementation of GPT-4o-mini for named entity recognition
"""
import os
import json
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from gptner import GPT4oNER

def test_k_values(average: int = 3, outfile: str = "gptner_kvalues.json"):

    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    if os.path.isfile(outfile):
        with open(outfile, "r", encoding='utf-8') as f:
            data = json.load(f)

    REMAINING_STRATEGIES = [s for s in ["sentence", "random"] if ((s not in data) or (len(data[s].keys()) != average))]
    print(REMAINING_STRATEGIES)

    if REMAINING_STRATEGIES:
        model = GPT4oNER()
        for strategy in REMAINING_STRATEGIES:
            model.fewshot_strategy = strategy
            if strategy not in data:
                data[strategy] = {}

            for k in [1, 2, 4, 8, 16, 32, 64]:
                print(f"Testing with {k} few-shot examples")
                remaining_seeds = [str(s) for s in range(1, average + 1) if str(s) not in data[strategy] or str(k) not in data[strategy][str(s)]]
                for seed in remaining_seeds:
                    if seed not in data[strategy]:
                        data[strategy][seed] = {}
                    print(f"Testing model with {k} sentence embedding samples and seed {seed} using {strategy} few-shot strategy")
                    model.num_examples = k
                    data[strategy][seed].update({k: model.evaluate(sample=100, seed=int(seed))})

        with open(outfile, "w+", encoding='utf-8') as f:
            json.dump(data, f)

    fig, ax = plt.subplots()

    for strategy in data:
        ax.plot([int(k) for k in data[strategy]["1"].keys()], np.mean([list(data[strategy][str(seed)].values()) for seed in range(1, average + 1)], axis=0) * 100, marker=".")
        ax.grid(True, 'major')
        ax.set_xlabel("$k$")
        ax.set_ylabel("F1 score")
        ax.set_yticks(np.arange(50, 101, 10))
        ax.set_xticks([int(k) for k in data[strategy]["1"].keys()])
    fig.legend(["Sentence-level Embedding", "Random Sample"])
    fig.savefig("kvalues_1-32.png")
    plt.show()


def test_example_order(average: int = 3, outfile: str = "gptner_examples.json"):

    STRATEGIES = ["first", "last", "random"]
    data = {s: [] for s in STRATEGIES}

    if os.path.isfile(outfile):
        with open(outfile, "r", encoding='utf-8') as f:
            data = {**data, **json.load(f)}

    TO_TEST = [s for s in STRATEGIES if not data[s]]

    if TO_TEST:

        model = GPT4oNER(num_examples=32)

        for seed in range(1, average + 1):
            for strategy in TO_TEST:
                print(f"Testing ordering strategy {strategy} with seed {seed}")
                data[strategy].append(model.evaluate(sample=100, seed=seed, example_ordering=strategy))

        with open(outfile, "w+", encoding='utf-8') as f:
            json.dump(data, f)


    plt.bar(["Best $k$ First", "Best $k$ Last", "Random"], [np.mean(data["first"]), np.mean(data["last"]), np.mean(data["random"])])
    plt.xlabel("Example Ordering Strategy")
    plt.ylabel("F1 Score")
    plt.savefig("kordering_flr.png")
    plt.show()


if __name__ == "__main__":
    #test_example_order(average=3, outfile="gptner_examples_system.json")

    model = GPT4oNER(num_examples=32)

    data = []

    model.SYSTEM_PROMPT = "I am an excelent linguist. The task is to label location (@@), organisation (~~), persons (¬¬) and miscellaneous (>>) entities in the given sentence, signifying the end of an entity with ##. Miscellaneous entities are named entities that are not locations, persons or organisations and include (but are not limited to) events, nationalities, products, works of art. I must try to avoid identifying entities which I am not certain of as a false positive carries more weight than a false negative. Below are some examples.\n\n"
    print(f"F1: {model.evaluate(produce_report=True, output_file='proompt.json')}")

    print(data)
