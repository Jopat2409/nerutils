from local_datasets.loader import load_to_huggingface, Dataset
from transformers import AutoModelForTokenClassification
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast

class FLERTDataset(Dataset):

    def __init__(self, split: Dataset, tokenizer: XLMRobertaTokenizerFast):
        self.raw_dataset = split
        self.tokenizer = tokenizer


        tags, contexts = [], []
        for document in set(self.raw_dataset["document"]):

            print(f"Document {document}/{len(set(self.raw_dataset['document']))}", end='\r')

            examples = self.raw_dataset.filter(lambda example: example["document"] == document)

            encoding = self.tokenizer(examples["tokens"], return_offsets_mapping=True, is_split_into_words=True)
            assert len(encoding["input_ids"]) == len(examples)

            document_tokens_flat = [_id for sentence in encoding["input_ids"] for _id in sentence]
            offset = 0

            for i, sentence_tokens in enumerate(encoding["input_ids"]):
                start, end = offset, offset + len(sentence_tokens)

                left_ctx_len = min(64, start)
                right_ctx_len = min(64, len(document_tokens_flat) - end)

                context = document_tokens_flat[start-left_ctx_len:end+right_ctx_len]

                word_ids = encoding.word_ids(batch_index=i)
                aligned_labels = []
                prev_word_id = None

                for word_id in word_ids:
                    if word_id is None:
                        aligned_labels.append(-100)
                    elif word_id != prev_word_id:
                        aligned_labels.append(examples[i]["ner_tags"][word_id])
                    else:
                        aligned_labels.append(-100)  # Only label first subword
                    prev_word_id = word_id
                ner_tags = [-100 for _ in range(left_ctx_len)] + aligned_labels + [-100 for _ in range(right_ctx_len)]

                assert len(context) == len(ner_tags)

                tags.append(ner_tags)
                contexts.append(context)

                offset += len(sentence_tokens)
        
        self.raw_dataset.add_column("labels", tags)
        self.raw_dataset.add_column("context", )
        
        assert "ner_tags" in self.raw_dataset.features
        assert "tokens" in self.raw_dataset.features

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        example_document = self.raw_dataset[idx]["document"]
        print(example_document)
        return self.raw_dataset[idx]

class XlmRoberta:

    def __init__(self) -> None:

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
        self.model: XLMRobertaForTokenClassification = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large")

    def train(self, train: Dataset, valid: Dataset) -> None:

        train_ctx, validate_ctx = FLERTDataset(train, self.tokenizer), FLERTDataset(valid, self.tokenizer)

        print(train_ctx[10])


    def tokenize_and_align(self, examples):
        tokenized = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512, padding="max_length")
        return {"sdasd": tokenized}
        """ labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_idx = None
            label_ids = [] """

if __name__ == "__main__":

    global _tqdm_active
    _tqdm_active = False

    r = XlmRoberta()

    r.train(load_to_huggingface("local_datasets/conll/conll_03/train.txt"), load_to_huggingface("local_datasets/conll/conll_03/dev.txt"))
