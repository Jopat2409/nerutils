import os
import json
import random

from typing import Literal
from dataclasses import dataclass, fields

import torch
import seqeval
import seqeval.metrics

from tqdm import tqdm
from openai import OpenAI
from sklearn import metrics
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, combine, Dataset, DatasetDict
from transformers import LukeTokenizer, LukeForEntitySpanClassification, LukeConfig

@dataclass
class NERParams:
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    system_prompt: str = "I am an excellent linguist. The task is to label location (@@), organisation (~~), persons (¬¬) and miscellaneous (>>) entities in the given sentence, signifying the end of an entity with ##. Below are some examples.\n\n"

    sentence_embedding_model: str = "all-MiniLM-L6-v2"
    word_embedding_model: str = "studio-ousia/luke-large-finetuned-conll-2003"

    fewshot_examples: int = 32
    fewshot_strategy: Literal["random", "sentence", "word"] = "sentence"
    fewshot_ordering: Literal["random", "best_first", "best_last"] = "best_first"

    min_examples: int = 3
    similarity_pad: bool = False
    similarity_cutoff: float = 0.35
    use_similarity_cutoff: bool = False

    max_retries: int = 6


class LLMForNER:

    TOKENS = {"LOC": "@@", "ORG": "~~", "PER": "¬¬", "MISC": ">>"}
    VALUES = {v: k for k, v in TOKENS.items()}

    params: NERParams

    def __init__(self, api_key: str, train_dataset: str = "conll2003", **kwargs) -> None:
        """Create a GPT-based named entity recognition system

        Args:
            api_key (str): Your API key for the model you are using (OPENAI_API_KEY for GPT models, DEEPSEEK_API_KEY for deepseek models etc)
            train_dataset (str | DatasetDict): Either the name of a dataset hosted on Hugging Face or a Hugging Face Dataset Dict that has been loaded manually. The dataset must have text and ner_tags features
            **kwargs: Optional parameters for the NERParams
        """
        self.params = NERParams(
            **{
                arg: v
                for arg, v in kwargs.items()
                if arg in [f.name for f in fields(NERParams)]
            }
        )
        self.client = OpenAI(api_key=api_key, base_url=self.params.llm_base_url)

        if isinstance(train_dataset, str):
            conll = load_dataset(train_dataset, trust_remote_code=True)
        elif isinstance(train_dataset, (Dataset, DatasetDict)):
            conll = train_dataset
        else:
            raise TypeError("train_dataset must be of type str or DatasetDict")
        self.id_to_label = conll["train"].features["ner_tags"].feature.int2str
        self.label_to_id = conll["train"].features["ner_tags"].feature.str2int

        # Since we are not training we can use both train and validation data as examples
        self.reference_data: Dataset = combine.concatenate_datasets(
            [conll["train"], conll["validation"]]
        )

        self.reference_data = self.reference_data.map(self.preprocess_example)

        if self.params.fewshot_strategy == "sentence":
            self.load_sentence_embeddings()
            self.sentence_embeddings.to("cuda")
        if self.params.fewshot_strategy == "word":
            self.load_word_embeddings()

    def load_sentence_embeddings(self) -> None:
        """Create the sentence embeddings for the kNN few-shot selection strategy

        Loads embeddings from sentence_embeddings.pt if it exists, else calculates an embedding for every
        sentence in the loaded train + validation dataset
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sentence_transformer = SentenceTransformer(
            self.params.sentence_embedding_model, device=device
        )

        if os.path.isfile("sentence_embeddings.pt"):
            self.sentence_embeddings = torch.load("sentence_embeddings.pt")
            assert len(self.sentence_embeddings) == len(
                self.reference_data
            ), "Wrong number of embeddings for test dataset"
            print("Loaded sentence embeddings from cache file")
        else:
            self.sentence_embeddings = self.sentence_transformer.encode(
                self.reference_data["text"], convert_to_tensor=True
            )
            print(
                "Generated sentence embeddings for all sentences, saved to sentence_embeddings.pt"
            )
            torch.save(self.sentence_embeddings, "sentence_embeddings.pt")

    def get_representation(
        self, text: str, ents: list[int] | None = None
    ) -> torch.Tensor:
        tokenized = self.word_tokenizer(text, return_tensors="pt")

        tokenized.to("cuda")
        if ents:
            ents = [self.id_to_label(i) for i in ents]
            word_ids = [
                self.word_tokenizer.convert_ids_to_tokens(tok.item())
                for tok in tokenized["input_ids"][0]
            ]
            tokens = [
                i for i, word in enumerate(word_ids) if word.startswith("Ġ") or i == 1
            ]
            ent_indices = [
                tidx for i, tidx in enumerate(tokens) if ents[i].startswith("B")
            ]
            assert len(tokens) == len(ents), f"{text} ({word_ids}): {ents}"

        representations = self.word_transformer.luke.forward(
            **tokenized
        ).last_hidden_state[0]
        if ents:
            return representations[ent_indices]
        return representations

    def create_word_embedding_datastore(self) -> None:
        self.word_embeddings = []

        dataloader = DataLoader(
            [self.reference_data["text"], self.reference_data["ner_tags"]], batch_size=8
        )
        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Generating word embeddings for dataset"
            ):
                self.word_embeddings += [
                    self.get_representation(sentence, ents) for sentence, ents in batch
                ]

        torch.save(self.word_embeddings, "word_embeddings.pt")

    def load_word_embeddings(self) -> None:
        print("Loading word embeddings")
        self.word_tokenizer: LukeTokenizer = LukeTokenizer.from_pretrained(
            "studio-ousia/luke-large"
        )
        self.word_transformer = LukeForEntitySpanClassification.from_pretrained(
            self.params.word_embedding_model,
            config=LukeConfig.from_pretrained(self.params.word_embedding_model),
        )
        self.word_transformer.to("cuda")

        if os.path.isfile("word_embeddings.pt"):
            self.word_embeddings = torch.load("word_embeddings.pt")
            assert len(self.word_embeddings) == len(
                self.reference_data
            ), "Data not aligned properly"
            print("Loaded word embeddings from file")
        else:
            self.create_word_embedding_datastore()

        sentence_lengths = [t.shape[0] for t in self.word_embeddings]

        self.word_to_sentence = []
        for i, length in enumerate(sentence_lengths):
            self.word_to_sentence += [i for i in range(length)]

        self.word_embeddings = torch.cat(self.word_embeddings, dim=0)

    def prompt_model(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.params.llm_model,
                messages=[
                    {"role": "system", "content": self.params.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Could not fetch {prompt}")
            print(e)
        return ""

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

    def get_examples(self, sentence: str) -> str:
        if self.params.fewshot_strategy == "random":
            examples = self.reference_data.shuffle().select(
                range(self.params.fewshot_examples)
            )
        elif self.params.fewshot_strategy == "word":
            rep = self.get_representation(sentence)
            sim = torch.cdist(rep, self.word_embeddings)
            flat_sim = sim.view(-1)
            _, topk_indices = torch.topk(
                flat_sim, k=self.params.fewshot_examples, largest=False
            )
            dataset_indices = topk_indices % sim.shape[1]

            examples = self.reference_data[
                [self.word_to_sentence[i] for i in dataset_indices]
            ]
        else:
            sim = self.sentence_transformer.similarity(
                self.sentence_embeddings,
                self.sentence_transformer.encode(
                    (sentence,), device="cuda", convert_to_tensor=True
                ),
            )
            flat_sim = sim.flatten()
            topk_values, topk_indices = flat_sim.topk(k=self.params.fewshot_examples)
            if self.params.use_similarity_cutoff:
                top_k_cutoff = topk_indices[topk_values > self.params.similarity_cutoff]
                examples: torch.Tensor = self.reference_data[top_k_cutoff] if top_k_cutoff.numel() else self.reference_data[topk_indices[0:self.params.min_examples]]
            else:
                examples = self.reference_data[topk_indices]
        # reverse list if we want the best k last

        example_strings = examples["fs_example"].copy()
        while len(example_strings) < self.params.fewshot_examples and self.params.similarity_pad and self.params.use_similarity_cutoff:
                example_strings += example_strings
        example_strings = example_strings[0:self.params.fewshot_examples]

        if self.params.fewshot_ordering == "random":
            random.shuffle(example_strings)
        elif self.params.fewshot_ordering == "last":
            example_strings = example_strings[::-1]

        return "\n\n".join(example_strings)

    def construct_prompt(self, example) -> None:
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
        dataset: str | Dataset = "conll2003",
        entity_wise: bool = False,
        produce_report: bool = False,
        sample: int = -1,
        seed: int = 1,
    ) -> float:
        """Given a dataset, evaluate the ability of this model to perform named entity recognition

        Args:
            output_file (str, optional): What file to write the output to. Defaults to no file "".
            dataset (str, optional): What dataset to evaluate. Defaults to "conll2003".
            entity_wise (bool, optional): Whether to prompt the model individually for each entity or combine into one prompt
            produce_report (bool, optional): Whether to print the full classification report. Defaults to False.

        Returns:
            str: _description_
        """

        if isinstance(dataset, str):
            test_data = load_dataset(dataset, trust_remote_code=True, split="test")
        else:
            test_data = dataset

        if sample != -1:
            test_data = test_data.shuffle(seed=seed).select(range(sample))

        total = len(test_data)

        data = []
        for i, example in enumerate(test_data):
            if self.params.fewshot_examples:
                sys_prompt = self.SYSTEM_PROMPT + self.get_examples(
                    " ".join(example["tokens"])
                )
            else:
                sys_prompt = self.SYSTEM_PROMPT.replace(" Below are some examples.", "")
            output = (
                self.prompt_model(
                    self.construct_prompt(example), system_prompt=sys_prompt
                )
                .replace("Output:", "")
                .lstrip()
            )

            isolated_output = self.isolate_output(output)
            aligned_output = self.align_model_output(example["tokens"], isolated_output)

            data.append({**example, "prediction": aligned_output})
            if i % 50 == 0:
                print(f"Predicted {i} / {total}", end="\r")

            if i % 500 == 0 and output_file:
                with open(output_file, "w+", encoding="utf-8") as f:
                    json.dump(data, f)

        if output_file:
            with open(output_file, "w+", encoding="utf-8") as f:
                json.dump(data, f)

        f1 = self.evaluate_f1(
            data,
            print_report=produce_report,
            id2label=test_data.features["ner_tags"].feature.int2str,
        )
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

    def evaluate_f1(
        self, examples: list[dict], print_report: bool, view_confusion: bool = False, id2label: callable = None
    ) -> float:
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
        tokens = []

        incorrect = 0
        for example in examples:
            gold_labels = [
                id2label(tag) if id2label is not None else self.id_to_label(tag)
                for tag in example["ner_tags"]
            ]

            prediction = self.isolate_output(example["prediction"])
            prediction = self.align_model_output(example["tokens"], prediction)

            predicted_labels = self.output_to_tags(prediction)

            if len(predicted_labels) == len(gold_labels):
                gold.append(gold_labels)
                predicted.append(predicted_labels)
                tokens.append(example["tokens"])
            else:
                incorrect += 1

        print(f"{incorrect} incorrectly formatted outputs from the model")

        if print_report:
            print(seqeval.metrics.classification_report(gold, predicted, digits=4))

        if view_confusion:
            gold_flattened = [tag[2:] if tag != "O" else tag for sentence in gold for tag in sentence]
            pred_flattened = [tag[2:] if tag != "O" else tag for sentence in predicted for tag in sentence]
            print(f"{float(len([i for i, g in enumerate(gold_flattened) if g == 'O' and pred_flattened[i] != 'O'])) / len([i for i, g in enumerate(gold_flattened) if g != pred_flattened[i]]) * 100}%")
            tokens_flattened = [token for sentence in tokens for token in sentence]

            assert len(gold_flattened) == len(pred_flattened) == len(tokens_flattened)

            with open("cleanconll.txt", "w+") as f:
                for g, p, t in zip(gold_flattened, pred_flattened, tokens_flattened):
                    f.write(f"{t} {g} {p}\n")


            confusion_matrix = metrics.confusion_matrix(gold_flattened, pred_flattened, normalize="pred")
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = sorted(set(gold_flattened)))
            print(set([tag for sentence in gold for tag in sentence]))
            fig, ax = plt.subplots(figsize=(10, 8))
            cm_display.plot(cmap="Blues", colorbar=False, ax=ax)
            plt.title("GPT + Sentence Embedding Examples")
            plt.tight_layout()
            plt.show()

        return seqeval.metrics.f1_score(gold, predicted)
