# Description

BERT sequence tagger that accepts token list as an input (not BPE but any "general" tokenizer like NLTK or Standford) and produces tagged results in IOB format.

Training BERT model has many caveats that include but not limited to:  
- Proper masking of the input.
- Proper padding of input.
- Loss masking (masking loss of the padded tokens and loss of the BPE suffixes).
- Adding proper special tokens like [CLS], [SEP] to the beginning and an end of a sequence.
- Annealing of the learning rate, as well as properly handling the best models.
- Proper calculation of the validation / training loss (taking into account masked tokens and masked loss elements).

Pytorch_transformers provides a good pytorch implementation of BertForTokenClassification (not wihtout flaws), however, it lacks code for proper trainig of sequence tagging models. Noticable effort is required to convert a tokenized text into an input suitable for BERT, with which you can achieve SOTA.

This library does this work for you: it takes a tokenized input, performs bpe tokenization, padding, preparations, and all other work to prepare input for BERT. It also provides a trainer that can achieve a good performance for BERT models. 

# Example

```python

from sequence_tagger_bert import SequenceTaggerBert, ModelTrainerBert
from flair.datasets import ColumnCorpus

data_folder = '../data/conll2003'

corpus = ColumnCorpus(data_folder, 
                      {0 : 'text', 3 : 'ner'},
                      train_file='eng.train.txt',
                      test_file='eng.testb.txt',
                      dev_file='eng.testa.txt')

bpe_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', 
                                              do_lower_case=False)

model, optimizer = create_model_optimizer(tag2idx, 
                                          full_finetuning=True, 
                                          base_lr=5e-5,
                                          bert_model='bert-base-cased')

```