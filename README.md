# Description

BERT sequence tagger that accepts token list as an input (not BPE but any "general" tokenizer like NLTK or Standford) and produces tagged results in IOB format.

Training BERT model has many caveats that include but not limited to:  
- Proper masking of the input.
- Proper padding of input.
- Loss masking (masking loss of the padded tokens and loss of the BPE suffixes).
- Adding proper special tokens like [CLS], [SEP] to the beginning and an end of a sequence.
- Annealing of the learning rate, as well as properly handling the best models.
- Proper calculation of the validation / training loss (taking into account masked tokens and masked loss elements).

[Pytorch_transformers](https://github.com/huggingface/transformers) provides a good pytorch implementation of BertForTokenClassification, however, it lacks code for proper trainig of sequence tagging models. Noticable effort is required to convert a tokenized text into an input suitable for BERT, with which you can achieve SOTA.

This library does this work for you: it takes a tokenized input, performs bpe tokenization, padding, preparations, and all other work to prepare input for BERT. It also provides a trainer that can achieve the best performance for BERT models. See below example for CoNLL-2003 dataset. More detailed example in jupyter notebook is [here](http://github/iinemo/bert_for_sequence_tagging/src/example.ipynb).

# Example

```python

from bert_sequence_tagger import SequenceTaggerBert, BertForTokenClassificationCustom
from pytorch_transformers import BertTokenizer

from bert_sequence_tagger.bert_utils import get_model_parameters, prepare_flair_corpus
from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus 
from bert_sequence_tagger.model_trainer_bert import ModelTrainerBert
from bert_sequence_tagger.metrics import f1_entity_level, f1_token_level

from pytorch_transformers import AdamW, WarmupLinearSchedule


import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('sequence_tagger_bert')


# Loading corpus ############################

from flair.datasets import ColumnCorpus

data_folder = './conll2003'
corpus = ColumnCorpus(data_folder, 
                      {0 : 'text', 3 : 'ner'},
                      train_file='eng.train',
                      test_file='eng.testb',
                      dev_file='eng.testa')


# Creating model ############################

batch_size = 16
n_epochs = 4
model_type = 'bert-base-cased'
bpe_tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=False)

idx2tag, tag2idx = make_bert_tag_dict_from_flair_corpus(corpus)

model = BertForTokenClassificationCustom.from_pretrained(model_type, 
                                                         num_labels=len(tag2idx)).cuda()

seq_tagger = SequenceTaggerBert(bert_model=model, bpe_tokenizer=bpe_tokenizer, 
                                idx2tag=idx2tag, tag2idx=tag2idx, max_len=128,
                                batch_size=batch_size)


# Training ############################

train_dataset = prepare_flair_corpus(corpus.train)
val_dataset = prepare_flair_corpus(corpus.dev)

optimizer = AdamW(get_model_parameters(model), lr=5e-5, betas=(0.9, 0.999), 
                  eps=1e-6, weight_decay=0.01, correct_bias=True)

n_iterations_per_epoch = len(corpus.train) / batch_size
n_steps = n_iterations_per_epoch * n_epochs
lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0.1, t_total=n_steps)

trainer = ModelTrainerBert(model=seq_tagger, 
                           optimizer=optimizer, 
                           lr_scheduler=lr_scheduler,
                           train_dataset=train_dataset, 
                           val_dataset=val_dataset,
                           validation_metrics=[f1_entity_level],
                           batch_size=batch_size)

trainer.train(epochs=n_epochs)

# Testing ############################

test_dataset = prepare_flair_corpus(corpus.test)
_, __, test_metrics = seq_tagger.predict(test_dataset, evaluate=True, 
                                         metrics=[f1_entity_level, f1_token_level])
print(f'Entity-level f1: {test_metrics[1]}')
print(f'Token-level f1: {test_metrics[2]}')

# Predicting
seq_tagger.predict([['We', 'are', 'living', 'in', 'New', 'York', 'city', '.']])

```

# Installation

pip install git+https://github.com/IINemo/bert_sequence_tagger.git

# Requirements

- torch
- pytorch-transformers
- flair
- seqeval

# Cite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@inproceedings{shelmanov2019bibm,
    title={Active Learning with Deep Pre-trained Models for Sequence Tagging of Clinical and Biomedical Texts},
    author={Artem Shelmanov and Vadim Liventsev and Danil Kireev and Nikita Khromov and Alexander Panchenko and Irina Fedulova and Dmitry V. Dylov},
    booktitle={Proceedings of International Conference on Bioinformatics & Biomedicine (BIBM)},
    year={2019}
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~