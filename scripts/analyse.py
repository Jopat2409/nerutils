import sys
import seqeval

import matplotlib.pyplot as plt
from sklearn import metrics

if __name__ == "__main__":
    _, infile, fmt = sys.argv
    assert fmt in ["conll", "json"]

    gold, pred = [], []
    if fmt == "conll":
        with open("predictions.txt", "r") as f:
            for line in f.readlines():
                if line.startswith("-DOCSTART-") or not line.strip():
                    continue
                _, g, p = line.strip().split(" ")
                gold.append(g)
                pred.append(p)

        print(seqeval.metrics.classification_report([gold,], [pred,], digits=4))

        confusion_matrix = metrics.confusion_matrix(gold, pred, normalize="pred")
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = sorted(set(gold + pred)))

        fig, ax = plt.subplots(figsize=(10, 8))
        cm_display.plot(cmap="Blues", colorbar=False, ax=ax)
        plt.title("XLM-RoBERTA trained for 5 epochs on CleanCoNLL")
        plt.tight_layout()
        plt.savefig("xlmr-cleanconll5.png")
        plt.show()
    elif fmt == "json":
        
