import seqeval
import seqeval.metrics
import torch
from flair.datasets import CONLL_03, On
from flair.models import SequenceTagger
import matplotlib.pyplot as plt

from sklearn import metrics




if __name__ == "__main__":
    """ corpus = CONLL_03(base_path="..\\local_datasets\\cleanconll\\")

    tagger = SequenceTagger.load('..\\models\\xlm-cleanconll-5epochs.pt')
    tagger.to('cuda')

    result = tagger.evaluate(corpus.test, 'ner', out_path="cc5predictions.txt") """

    gold, pred, tokens = [], [], []
    with open("predictions.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("-DOCSTART-") or not line.strip():
                continue
            t, g, p = line.strip().split(" ")
            tokens.append(t)
            gold.append(g[2:] if g != "O" else g)
            pred.append(p[2:] if p != "O" else p)

    with open("predictions_normalized.txt", "w+") as f:
        for t, g, p in zip(tokens, gold, pred):
            f.write(f"{t} {g} {p}\n")

    print(seqeval.metrics.classification_report([gold,], [pred,], digits=4))

    confusion_matrix = metrics.confusion_matrix(gold, pred, normalize="pred")
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = sorted(set(gold + pred)))

    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display.plot(cmap="Blues", colorbar=False, ax=ax)
    plt.title("XLM-RoBERTA CoNLL-2003")
    plt.tight_layout()
    plt.show()
