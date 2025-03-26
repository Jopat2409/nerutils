import logging

from transformers import LukeConfig, LukeTokenizer, LukeForEntitySpanClassification
from datasets import load_dataset


class LukeTrainer:
    logger: logging.Logger

    config: LukeConfig
    tokenizer: LukeTokenizer
    model: LukeForEntitySpanClassification

    def __init__(
        self,
        model: str = "studio-ousia/luke-base",
        label_scheme="IOB2",
        dataset="conll2003",
        max_entity_length: int = 128,
        max_mention_length: int = 32,
        default_non_entity: str = "O",
    ) -> None:
        self.base_model = model

        self.dataset = dataset
        self.label_scheme = label_scheme
        self.default_non_entity = default_non_entity

        self.max_entity = max_entity_length
        self.max_mention = max_mention_length

        self.setup_logging()
        self.load_dataset()
        self.create_model()

        self.logger.info(self.model.config.label2id)
        self.logger.info(self.model.config.id2label)

    def resolve_id2label(self) -> None:
        """
        Most pre-trained models have their id2label and label2id dictionaries all wrong.
        Lets update that based on the dataset we are using!

        Since LUKE does span classification, if the data is in IOB format we need to remove the I- and B- from the labels
        """
        labels = ["O"] + sorted(
            list(
                set(
                    label.replace("B-", "").replace("I-", "")
                    for label in self.raw_dataset["train"]
                    .features["ner_tags"]
                    .feature.names
                    if label != self.default_non_entity
                )
            )
        )
        self.config.label2id = {label: i for i, label in enumerate(labels)}
        self.config.id2label = {i: label for label, i in self.config.label2id.items()}

    def setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        self.logger = logging.getLogger(__name__)

    def preprocess_dataset(self, examples) -> None:
        print(examples)

    def load_dataset(self) -> None:
        """Load the dataset from the config"""
        self.raw_dataset = load_dataset(self.dataset, trust_remote_code=True)

        self.raw_dataset.map(
            lambda x: self.preprocess_dataset(self, x),
            desc="Preprocessing dataset",
            batched=True,
            remove_columns=self.raw_dataset["train"].column_names,
        )

    def create_model(self) -> None:
        """Generate the model, config and tokenizer from the settings
        currently stored in this class
        """

        self.config = LukeConfig.from_pretrained(self.base_model)
        self.resolve_id2label()

        self.tokenizer = LukeTokenizer.from_pretrained(
            self.base_model,
            task="entity_span_classification",
            max_entity_length=self.max_entity,
            max_mention_length=self.max_mention,
        )

        self.model = LukeForEntitySpanClassification.from_pretrained(
            self.base_model, config=self.config
        )


if __name__ == "__main__":
    trainer = LukeTrainer()
