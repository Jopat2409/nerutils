"""
Functions used to evaluate and test the implementation of GPT-4o-mini for named entity recognition
"""
import os
import json
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from datasets import DatasetDict

from local_datasets.loader import load_to_huggingface

from llm import LLMForNER

def test_k_values(average: int = 3, outfile: str = "gptner_kvalues.json"):

    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    if os.path.isfile(outfile):
        with open(outfile, "r", encoding='utf-8') as f:
            data = json.load(f)

    REMAINING_STRATEGIES = [s for s in ["sentence", "random"] if ((s not in data) or (len(data[s].keys()) != average))]
    print(REMAINING_STRATEGIES)

    if REMAINING_STRATEGIES:
        for strategy in REMAINING_STRATEGIES:
            model = LLMForNER(fewshot_strategy=strategy)
            if strategy not in data:
                data[strategy] = {}

            for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                remaining_seeds = [str(s) for s in range(1, average + 1) if str(s) not in data[strategy] or str(k) not in data[strategy][str(s)]]

                if not remaining_seeds:
                    continue

                print(f"Testing with {k} few-shot examples")
                for seed in remaining_seeds:
                    if seed not in data[strategy]:
                        data[strategy][seed] = {}
                    print(f"Testing model with {k} sentence embedding samples and seed {seed} using {strategy} few-shot strategy")
                    model.params.fewshot_examples = k
                    data[strategy][seed].update({k: model.evaluate(sample=100, seed=int(seed))})

        with open(outfile, "w+", encoding='utf-8') as f:
            json.dump(data, f)

    fig, ax = plt.subplots()

    for strategy in data:
        ax.plot(np.log2([int(k) for k in data[strategy]["1"].keys()]), np.mean([list(data[strategy][str(seed)].values()) for seed in range(1, average + 1)], axis=0) * 100, marker=".")
        ax.grid(True, 'major')
        ax.set_xlabel("$log_2(k)$")
        ax.set_ylabel("F1 score")
        ax.set_yticks(np.arange(50, 101, 10))
        ax.set_xlabels([int(k) for k in data[strategy]["1"].keys()])
        #ax.set_xticks([int(k) for k in data[strategy]["1"].keys()])
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

        model = LLMForNER(num_examples=32)

        for seed in range(1, average + 1):
            for strategy in TO_TEST:
                print(f"Testing ordering strategy {strategy} with seed {seed}")
                data[strategy].append(model.evaluate(sample=100, seed=seed, example_ordering=strategy))

        with open(outfile, "w+", encoding='utf-8') as f:
            json.dump(data, f)


    plt.bar(["Best $k$ First", "Best $k$ Last", "Random"], [np.mean(data["first"])*100, np.mean(data["last"])*100, np.mean(data["random"])*100])
    plt.xlabel("Example Ordering Strategy")
    plt.ylabel("F1 Score")
    plt.savefig("kordering_flr.png")
    plt.show()

def test_prompt(prompt: str):
    model = LLMForNER(num_examples=32)
    DEFAULT_PROMPT = model.SYSTEM_PROMPT

    default = 0
    tested = 0

    for seed in [1,2,3]:
        default += model.evaluate(sample=100, seed=seed, produce_report=True)
        model.SYSTEM_PROMPT = prompt
        tested += model.evaluate(sample=100, seed=seed, produce_report=True)
        model.SYSTEM_PROMPT = DEFAULT_PROMPT
    print(f"Previous: {default / 3}, New: {tested / 3}")

def test_conllpp():
    train = DatasetDict({"train": load_to_huggingface("local_datasets/cleanconll/train.txt"), "validation": load_to_huggingface("local_datasets/cleanconll/dev.txt")})
    print(train["train"].features["ner_tags"])
    print(train["validation"].features["ner_tags"])

    model = LLMForNER(fewshot_examples = 32, train_dataset=train)
    conllpp = load_to_huggingface("local_datasets/conllpp/test.txt")
    model.evaluate(produce_report=True, output_file='conllpptest_cleanconll.json', dataset=conllpp)

def test_cleanconll():
    train = DatasetDict({"train": load_to_huggingface("local_datasets/cleanconll/train.txt"), "validation": load_to_huggingface("local_datasets/cleanconll/dev.txt")})
    evaluate = load_to_huggingface("local_datasets/cleanconll/test.txt")

    model = LLMForNER(fewshot_examples = 32, train_dataset=train)
    model.evaluate(output_file="cleanconll2.json", produce_report=True, dataset=evaluate)

def test_bad_examples():

    model = LLMForNER(use_similarity_cutoff=True)

    TO_TEST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


    data = {"Padding": [], "No Padding": []}

    for seed in [1, 2, 3]:
        for policy in data:
            data[policy].append([])
            model.params.similarity_pad = policy == "Padding"
            for test in TO_TEST:
                print(f"Testing {policy} with {test} (seed {seed})")
                model.params.similarity_cutoff = test
                data[policy][-1].append(model.evaluate(sample=100, seed=seed))

    with open("bad_example_test.json", "w", encoding='utf-8') as f:
        data = json.dump(data, f)

    model.params.use_similarity_cutoff = False
    control = model.evaluate(sample=100, seed=1)

    fig, ax = plt.subplots()
    ax.axhline(control, linestyle='--')
    for strategy in data:
        ax.plot(TO_TEST, data[strategy])
        ax.grid(True, 'major')
        ax.set_xlabel("$Similarity Cutoff$")
        ax.set_ylabel("F1 score")
        #ax.set_xticks([int(k) for k in data[strategy]["1"].keys()])
    fig.legend(["Normal $k$-shot examples", "Padding to $k$", "No padding"])
    fig.savefig("badexamples.png")
    plt.show()
