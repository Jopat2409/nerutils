import argparse

import seqeval
import seqeval.metrics
from sklearn import metrics
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Produce detailed classification reports and confusion matrices for predictions')
    parser.add_argument('input', help="The prediction text file")

    parser.add_argument("-c", "--confusion", help="Produce confusion matrix", action="store_true")
    parser.add_argument("-t", "--title", help="Confusion Matrix Title", default="Confusion Matrix")
    parser.add_argument("-f", "--figure", help="Name of confusion matrix figure (or None)", default=None)
    parser.add_argument("-n", "--normalise", help="Normalise confusion matrix", default="pred")

    args = parser.parse_args()

    gold, pred = [], []
    with open(args.input, "r", encoding='utf-8') as f:
        for line in f.readlines():
            if line.startswith("-DOCSTART-") or not line.strip():
                continue
            _, g, p = line.strip().split(" ")
            gold.append(g[2:] if g != "O" else g)
            pred.append(p[2:] if p != "O" else p)

    print(seqeval.metrics.classification_report([gold,], [pred,], digits=4))

    if args.confusion:
        confusion_matrix = metrics.confusion_matrix(gold, pred, normalize=args.normalise)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = sorted(set(gold + pred)))

        fig, ax = plt.subplots(figsize=(10, 8))
        cm_display.plot(cmap="Blues", colorbar=False, ax=ax)
        plt.title(args.title)
        plt.tight_layout()

        if args.figure is not None:
            fig.savefig(args.figure)

        plt.show()
