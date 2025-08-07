import random

from flair.datasets import CONLL_03
from flair.data import FlairDataset
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = CONLL_03(base_path="..\\local_datasets\\conll\\")
corpus.downsample(downsample_test=False)

""" """ # 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True
)

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

tagger.to('cuda')

num_params = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")
