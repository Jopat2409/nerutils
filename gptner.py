import os
import json
import torch
import random

from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal

from datasets import load_dataset, combine, Dataset
from sentence_transformers import SentenceTransformer
import seqeval
import seqeval.metrics


class GPT4oNER:
    TOKENS = {"LOC": "@@", "ORG": "~~", "PER": "¬¬", "MISC": ">>"}
    VALUES = {v: k for k, v in TOKENS.items()}
    SYSTEM_PROMPT = "I am an excelent linguist. The task is to label location (@@), organisation (~~), persons (¬¬) and miscellaneous (>>) entities in the given sentence, signifying the end of an entity with ##. Below are some examples.\n\n"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        fewshot_strategy: Literal["random", "sentence"] = "sentence",
        num_examples: int = 10,
        max_retries: int = 6,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Create a GPT-based named entity recognition system

        Args:
            fewshot_strategy (Literal['random', 'sentence'], optional): The strategy to use for selecting the few-shot examples. Sentence for k-nearest-neighbour sentence-level embeddings, random for a random sample Defaults to 'sentence'.
            num_examples (int, optional): Number of few-shot examples to give. Defaults to 10.
            max_retries (int, optional): Max number of times to prompt the model for a valid output if the initial answer is of an invalid format. Defaults to 6.
            self_verification (bool, optional): Whether or not to perform self-verification. Defaults to false.
            model (str, optional): The OpenAI model to use. Defaults to 'gpt-4o-mini'.
        """
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.num_examples = num_examples
        self.max_retries = max_retries
        self.fewshot_strategy = fewshot_strategy
        self.gpt_model = model

        conll = load_dataset("conll2003", trust_remote_code=True)
        self.id_to_label = conll["test"].features["ner_tags"].feature.int2str

        # Since we are not training we can use both train and validation data as examples
        self.reference_data: Dataset = combine.concatenate_datasets(
            [conll["train"], conll["validation"]]
        )
        self.reference_data = self.reference_data.map(self.preprocess_example)

        self.sentence_transformer = SentenceTransformer(self.EMBEDDING_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_transformer = self.sentence_transformer.to(device)

        if os.path.isfile("sentence_embeddings.pt"):
            self.sentence_embeddings = torch.load("sentence_embeddings.pt")
            assert len(self.sentence_embeddings) == len(self.reference_data)
            print("Loaded sentence embeddings from cache file")
        else:
            self.sentence_embeddings = self.sentence_transformer.encode(
                self.reference_data["text"], convert_to_tensor=True
            )
            torch.save(self.sentence_embeddings, "sentence_embeddings.pt")
        print("Generated sentence embeddings for all sentences")

    def create_individual_system_prompt(self, entity: str) -> str:
        return f"I am an excellent linguist. Your task is to label {entity} entities (@@) in the given sentence, signifying the end of an entity with ##. Below are some examples.\n\n"

    def prompt_model(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def preprocess_example(self, example: dict) -> dict:
        input_text = " ".join(example["tokens"])

        labels = [self.id_to_label(lb) for lb in example["ner_tags"]]
        output_text = ""

        for i, word in enumerate(example["tokens"]):
            c, prev = labels[i], labels[i - 1] if i != 0 else "O"
            if (c == "O" or c.startswith("B")) and prev != "O":
                output_text += "##"
            output_text += " "
            if c.startswith("B"):
                output_text += self.TOKENS[c[2:]]
            output_text += word

        example["fs_example"] = f"Input: {input_text}\nOutput: {output_text}"
        example["text"] = input_text
        return example

    def get_examples(
        self, sentence: str, n: int, _random: bool = False, ordering: Literal['first', 'last', 'random'] = 'first'
    ) -> str:
        if _random:
            examples = self.reference_data.shuffle(seed=72).select(range(n))
        else:
            sim = self.sentence_transformer.similarity(
                self.sentence_embeddings, self.sentence_transformer.encode((sentence,))
            )
            flat_sim = sim.flatten()
            _, topk_indices = flat_sim.topk(k=n)
            examples = self.reference_data[topk_indices]
        # reverse list if we want the best k last

        example_strings = examples["fs_example"]

        if ordering == 'random':
            random.shuffle(example_strings)
        elif ordering == 'last':
            example_strings = example_strings[::-1]

        return "\n\n".join(example_strings)

    def construct_prompt(self, example, example_ordering: Literal['first', 'last', 'random']) -> None:
        input_text = " ".join(example["tokens"])
        return f"Input: {input_text}"

    def align_model_output(self, gold_tokens: list[str], model_output: str) -> str:
        """GPT has a tendency to occasionally miss certain tokens in the output string. Since these outputs
        are still formatted corretly the tokens can be added back in and it can be assumed that the model did not assign
        an entity type to them

        Args:
            gold_tokens (list[str]): _description_
            model_output (str): _description_

        Returns:
            str: _description_
        """

        if model_output.startswith("Output:"):
            model_output = model_output.replace("Output:", "").lstrip()

        output_tokens = model_output.split(" ")
        if len(output_tokens) == len(gold_tokens):
            return model_output

        REPLACE = list(self.VALUES.keys()) + ["##"]

        for i in range(len(gold_tokens)):
            if i >= len(output_tokens):
                output_tokens.append(gold_tokens[i])
            else:
                clean_output = output_tokens[i]
                for v in REPLACE:
                    clean_output = clean_output.replace(v, "")
                if gold_tokens[i] != clean_output:
                    output_tokens.insert(i, gold_tokens[i])
        return " ".join(output_tokens)

    def isolate_output(self, model_output: str) -> str:
        if "Output:" in model_output:
            model_output = model_output[model_output.find("Output:") :]
        return model_output

    def evaluate(
        self,
        output_file: str = "",
        dataset: str = "conll2003",
        entity_wise: bool = False,
        example_ordering: Literal['first', 'last', 'random'] = 'first',
        produce_report: bool = False,
        sample: int = -1,
        seed: int = 1,
    ) -> str:
        """Given a dataset, evaluate the ability of this model to perform named entity recognition

        Args:
            output_file (str, optional): What file to write the output to. Defaults to no file "".
            dataset (str, optional): What dataset to evaluate. Defaults to "conll2003".
            entity_wise (bool, optional): Whether to prompt the model individually for each entity or combine into one prompt
            produce_report (bool, optional): Whether to print the full classification report. Defaults to False.

        Returns:
            str: _description_
        """

        test_data = load_dataset(dataset, trust_remote_code=True, split="test")

        if sample != -1:
            test_data = test_data.shuffle(seed=seed).select(range(sample))

        total = len(test_data)

        data = []
        for i, example in enumerate(test_data):
            few_shot_examples = self.get_examples(
                " ".join(example["tokens"]),
                n=self.num_examples,
                _random=self.fewshot_strategy == "random",
                ordering=example_ordering,
            )
            output = (
                self.prompt_model(
                    self.construct_prompt(example, example_ordering),
                    self.SYSTEM_PROMPT + few_shot_examples
                )
                .replace("Output:", "")
                .lstrip()
            )

            isolated_output = self.isolate_output(output)
            aligned_output = self.align_model_output(example["tokens"], isolated_output)

            data.append({**example, "prediction": aligned_output})
            if i % 50 == 0:
                print(f"Predicted {i} / {total}", end="\r")
        if output_file:
            with open(output_file, "w+", encoding="utf-8") as f:
                json.dump(data, f)

        f1 = self.evaluate_f1(data, print_report=produce_report)
        return f1

    def output_to_tags(self, output_str: str) -> list:
        """Convert the output of the model to a sequence of tags in order to evaluate
        against the gold standard

        Args:
            output_str (str): _description_

        Returns:
            list: _description_
        """
        if output_str.startswith("Output: "):
            output_str = output_str.replace("Output:", "").lstrip()
        output_tokens = output_str.rstrip().split(" ")
        tags = []
        current_entity = "O"
        for token in output_tokens:
            if token[:2] in self.VALUES:
                current_entity = self.VALUES[token[:2]]
                tags.append(f"B-{current_entity}")
            else:
                tags.append(f"I-{current_entity}" if current_entity != "O" else "O")

            if token.endswith("##"):
                current_entity = "O"
        return tags

    def evaluate_f1(self, examples: list[dict], print_report: bool) -> float:
        """Given a dictioary containing tokens, gold label tags and model outputs,
        convert model outputs to labels and evaluate the entity-level F1 score
        for each entity type in the dataset

        Args:
            examples (list[dict]): _description_
            print_report (bool): Print the full classification report

        Returns:
            float: The F1 (micro averaged) score
        """
        gold = []
        predicted = []

        incorrect = 0
        for example in examples:
            gold_labels = [self.id_to_label(tag) for tag in example["ner_tags"]]

            prediction = self.isolate_output(example["prediction"])
            prediction = self.align_model_output(example["tokens"], prediction)

            predicted_labels = self.output_to_tags(prediction)

            if len(predicted_labels) == len(gold_labels):
                gold.append(gold_labels)
                predicted.append(predicted_labels)
            else:
                incorrect += 1

        print(f"{incorrect} incorrectly formatted outputs from the model")

        if print_report:
            print(seqeval.metrics.classification_report(gold, predicted))
        return seqeval.metrics.f1_score(gold, predicted)


if __name__ == "__main__":
    model = GPT4oNER(num_examples=1)
    # model.evaluate("gpt_10semb.json", "conll2003", True)

    print(f"F1: {model.evaluate(sample=100)}")
    # print(model.get_examples("SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT", 5))
