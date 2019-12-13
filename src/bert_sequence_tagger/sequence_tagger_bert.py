import torch
from torch.utils.data import DataLoader

from tensorflow.keras.preprocessing.sequence import pad_sequences

from .bert_for_token_classification_custom import BertForTokenClassificationCustom

import itertools
from tqdm import trange
import numpy as np
import pickle
import json
import os


import logging
logger = logging.getLogger('sequence_tagger_bert')


class SequenceTaggerBert:
    def __init__(self, bert_model, bpe_tokenizer, idx2tag, tag2idx, 
                 max_len=100, pred_loader_args={'num_workers' : 1}, 
                 pred_batch_size=100):
        super().__init__()
        
        self._bert_model = bert_model
        self._bpe_tokenizer = bpe_tokenizer
        self._idx2tag = idx2tag
        self._tag2idx = tag2idx
        self._max_len = max_len
        self._pred_loader_args = pred_loader_args
        self._pred_batch_size = pred_batch_size
        
    def _bpe_tokenize(self, words):
        new_words = []
        bpe_masks = []
        for word in words:
            bpe_tokens = self._bpe_tokenizer.tokenize(word)
            new_words += bpe_tokens        
            bpe_masks += [1] + [0] * (len(bpe_tokens) - 1)

        return new_words, bpe_masks
        
    def _make_tokens_tensors(self, tokens, max_len):
        bpe_tokens, bpe_masks = tuple(zip(*[self._bpe_tokenize(sent) for sent in tokens]))
        bpe_tokens = prepare_bpe_tokens_for_bert(bpe_tokens, max_len=max_len)
        bpe_masks = [[1] + masks[:max_len-2] + [1] for masks in bpe_masks]
        max_len = max(len(sent) for sent in bpe_tokens)
        token_ids = torch.tensor(create_tensors_for_tokens(self._bpe_tokenizer, bpe_tokens, max_len=max_len))
        token_masks = generate_masks(token_ids)
        return bpe_tokens, max_len, token_ids, token_masks, bpe_masks
    
    def _add_x_labels(self, labels, bpe_masks):
        result_labels = []
        for l_sent, m_sent in zip(labels, bpe_masks):
            m_sent = m_sent[1:-1]
            sent_res = []
            i = 0
            for l in l_sent:
                sent_res.append(l)
                
                i += 1
                while i < len(m_sent) and (m_sent[i] == 0):
                    i += 1
                    sent_res.append('[PAD]')
            
            result_labels.append(sent_res)
            
        return result_labels
    
    def _make_label_tensors(self, labels, bpe_masks, max_len):
        bpe_labels = self._add_x_labels(labels, bpe_masks)
        bpe_labels = prepare_bpe_labels_for_bert(bpe_labels, max_len=max_len)
        label_ids = torch.tensor(create_tensors_for_labels(self._tag2idx, bpe_labels, max_len=max_len))
        loss_masks = label_ids != self._tag2idx['[PAD]']
        return label_ids, loss_masks
    
    def _logits_to_preds(self, logits, bpe_masks, tokens):
        preds = logits.argmax(dim=2).numpy()
        probs = logits.numpy().max(axis=2)
        prob = [np.mean([p for p, m in zip(prob[:len(masks)], masks[:len(prob)]) if m][1:-1])  
                for prob, masks in zip(probs, bpe_masks)]
        preds = [[self._idx2tag[p] for p, m in zip(pred[:len(masks)], masks[:len(pred)]) if m][1:-1] 
                 for pred, masks in zip(preds, bpe_masks)]
        preds = [pred + ['O']*(max(0, len(toks) - len(pred))) for pred, toks in zip(preds, tokens)]
        return preds, prob
    
    def generate_tensors_for_prediction(self, evaluate, dataset_row):
        dataset_row = dataset_row
        labels = None
        if evaluate:
            tokens, labels = tuple(zip(*dataset_row))
        else:
            tokens = dataset_row
            
        _, max_len, token_ids, token_masks, bpe_masks = self._make_tokens_tensors(tokens, self._max_len)
        label_ids = None
        loss_masks = None
            
        if evaluate:
            label_ids, loss_masks = self._make_label_tensors(labels, bpe_masks, max_len)
        
        return token_ids, token_masks, bpe_masks, label_ids, loss_masks, tokens, labels
    
    def predict(self, dataset, evaluate=False, metrics=None):
        if metrics is None:
            metrics = []
        
        self._bert_model.eval()
        
        dataloader = DataLoader(dataset, 
                                collate_fn=lambda dataset_row: self.generate_tensors_for_prediction(evaluate, dataset_row), 
                               **self._pred_loader_args, 
                                batch_size=self._pred_batch_size)
        
        predictions = []
        probas = []
        
        if evaluate:
            cum_loss = 0.
            true_labels = []
                         
        for nb, tensors in enumerate(dataloader):
            token_ids, token_masks, bpe_masks, label_ids, loss_masks, tokens, labels = tensors

            if evaluate:
                true_labels.extend(labels)
            
            with torch.no_grad():
                token_ids = token_ids.cuda()
                token_masks = token_masks.cuda()
                
                if evaluate:
                    label_ids = label_ids.cuda()
                    loss_masks = loss_masks.cuda()
    
                if type(self._bert_model) is BertForTokenClassificationCustom:
                    logits = self._bert_model(token_ids, 
                                              token_type_ids=None,
                                              attention_mask=token_masks,
                                              labels=label_ids,
                                              loss_mask=loss_masks)
                else:
                    logits = self._bert_model(token_ids, 
                                              token_type_ids=None,
                                              attention_mask=token_masks,
                                              labels=label_ids,)
                
                if evaluate:
                    loss, logits = logits
                    cum_loss += loss.mean().item()
                else:
                    logits = logits[0]

                b_preds, b_prob = self._logits_to_preds(logits.cpu(), bpe_masks, tokens)
                
            predictions.extend(b_preds)
            probas.extend(b_prob)
                     
        if evaluate: 
            cum_loss /= (nb + 1)
            
            result_metrics = []
            for metric in metrics:
                result_metrics.append(metric(true_labels, predictions))
                     
            return predictions, probas, tuple([cum_loss] + result_metrics)
        else:
            return predictions, probas
        
    def generate_tensors_for_training(self, tokens, labels):
        _, max_len, token_ids, token_masks, bpe_masks = self._make_tokens_tensors(tokens, self._max_len)
        label_ids, loss_masks = self._make_label_tensors(labels, bpe_masks, max_len)
        return token_ids, token_masks, label_ids, loss_masks
    
    def generate_feature_tensors_for_prediction(self, tokens):
        _, max_len, token_ids, token_masks, bpe_masks = self._make_tokens_tensors(tokens, self._max_len)
        return token_ids, token_masks, bpe_masks

    def batch_loss_tensors(self, *tensors):
        token_ids, token_masks, label_ids, loss_masks = tensors
        token_ids = token_ids.cuda()
        token_masks = token_masks.cuda()
        label_ids = label_ids.cuda()
        loss_masks = loss_masks.cuda()
        
        if type(self._bert_model) is BertForTokenClassificationCustom:
            output = self._bert_model(token_ids, 
                                    token_type_ids=None,
                                    attention_mask=token_masks, 
                                    labels=label_ids,
                                    loss_mask=loss_masks)
        else:
            output = self._bert_model(token_ids, 
                                    token_type_ids=None, 
                                    attention_mask=token_masks, 
                                    labels=label_ids)
        
        loss = output[0]
        return loss.mean()
        
    def batch_loss(self, tokens, labels):
        token_ids, token_masks, label_ids, loss_masks = self.generate_tensors_for_training(tokens, labels)
        return self.batch_loss_tensors(token_ids, None, token_masks, label_ids, loss_masks)
    
    def batch_logits(self, tokens):
        _, max_len, token_ids, token_masks, __ = self._make_tokens_tensors(tokens, self._max_len)
        token_ids = token_ids.cuda()
        token_masks = token_masks.cuda()
        
        logits = self._bert_model(token_ids, 
                                token_type_ids=None,
                                attention_mask=token_masks, 
                                labels=None,
                                loss_mask=None)[0]

        return logits
    
    def save_serialize(self, save_dir_path):
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        
        torch.save(self._bert_model.state_dict(), os.path.join(save_dir_path, 'pytorch_model.bin'))
        with open(os.path.join(save_dir_path, 'bpe_tokenizer.pckl'), 'wb') as f:
            pickle.dump(self._bpe_tokenizer, f)
            
        self._bert_model.config.save_pretrained(os.path.join(save_dir_path))
        
        parameters_dict = {
            'idx2tag' : self._idx2tag,
            'tag2idx' : self._tag2idx,
            'max_len' : self._max_len,
            'pred_loader_args' : self._pred_loader_args,
            'pred_batch_size' : self._pred_batch_size
        }
        with open(os.path.join(save_dir_path, 'sec_parameters.json'), 'w') as f:
            json.dump(parameters_dict, f)

    @classmethod
    def load_serialized(cls, load_dir_path, bert_model_type):
        with open(os.path.join(load_dir_path, 'sec_parameters.json'), 'r') as f:
            parameters_dict = json.load(f)
         
        bert_model = bert_model_type.from_pretrained(load_dir_path).cuda()
        
        with open(os.path.join(load_dir_path, 'bpe_tokenizer.pckl'), 'rb') as f:
            bpe_tokenizer = pickle.load(f)
        
        return SequenceTaggerBert(bert_model, bpe_tokenizer,
                                  idx2tag=parameters_dict['idx2tag'], 
                                  tag2idx=parameters_dict['tag2idx'], 
                                  max_len=parameters_dict['max_len'], 
                                  pred_loader_args=parameters_dict['pred_loader_args'],
                                  pred_batch_size=parameters_dict['pred_batch_size'])
    
    # TODO: raw batch


def prepare_bpe_tokens_for_bert(tokens, max_len):
    return [['[CLS]'] + list(toks[:max_len - 2]) + ['[SEP]'] for toks in tokens]


def prepare_bpe_labels_for_bert(labels, max_len):
    return [['[PAD]'] + list(ls[:max_len - 2]) + ['[PAD]'] for ls in labels]


def generate_masks(input_ids):
    res = input_ids > 0
    return res.astype('float') if type(input_ids) is np.ndarray else res


def create_tensors_for_tokens(bpe_tokenizer, sents, max_len):
    return pad_sequences([bpe_tokenizer.convert_tokens_to_ids(sent) for sent in sents], 
                         maxlen=max_len, dtype='long', 
                         truncating='post', padding='post')


def create_tensors_for_labels(tag2idx, labels, max_len):
    return pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=max_len, value=tag2idx['[PAD]'], padding='post',
                         dtype='long', truncating='post')
