from typing import Any, List
from datasets import load_dataset

class NERSystem:

    def __init__(self) -> None:
        pass

    def output_to_tokens(self, model_output: Any) -> List[str]:
        "Each model must implement some method of converting "

        if isinstance(model_output, List):
            return model_output

        raise NotImplementedError("Models which do not output NER tags directly must implement a method to convert the output to tags")

    def preprocess(self, example: dict) -> dict:
        return example

    def evaluate(self, dataset: str, output_file: str, produce_report: bool = False):
        data = load_dataset(dataset, trust_remote_code=True, split="test")
        data.map(self.preprocess)

        for example in data:
            self.predict_example()