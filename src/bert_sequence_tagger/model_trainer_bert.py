import torch
from pytorch_transformers import AdamW

import copy
from tqdm import trange

import logging
logger = logging.getLogger('sequence_tagger_bert')


class ModelTrainerBert:
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler,
                 train_dataset, 
                 val_dataset, 
                 update_scheduler='es', # ee(every_epoch) or every_step(es)
                 keep_best_model=False,
                 restore_bm_on_lr_change=False,
                 max_grad_norm=1.0,
                 smallest_lr=0.,
                 validation_metrics=None,
                 decision_metric=None):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
            
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        
        self._update_scheduler = update_scheduler
        self._keep_best_model = keep_best_model
        self._restore_bm_on_lr_change = restore_bm_on_lr_change
        self._max_grad_norm = max_grad_norm
        self._smallest_lr = smallest_lr
        self._validation_metrics = validation_metrics
        self._decision_metric = decision_metric
        if self._decision_metric is None:
            self._decision_metric = lambda metrics: metrics[0]
    
    def train(self, epochs):
        best_model = {}
        best_dec_metric = float('inf')
        
        get_lr = lambda: self._optimizer.param_groups[0]['lr']
        
        train_dataloader = DataLoader(self._train_dataset, batch_size=self._batch_size, 
                                      collate_fn=self._model.generate_tensors_for_training)
        iterator = trange(epochs, desc='Epoch')
        for epoch in iterator:
            self._model._bert_model.train()

            cum_loss = 0.
            for nb, tensors in enumerate(train_dataloader):
                tensors = (t.cuda() for t in tensors)
                loss = self._model._bert_model(*tensors).mean()
                #loss = self._model.batch_loss(tokens, labels)
                cum_loss += loss.item()
                
                self._model._bert_model.zero_grad()
                loss.backward()
                if self._max_grad_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(parameters=self._model._bert_model.parameters(), 
                                                   max_norm=self._max_grad_norm)
                    
                self._optimizer.step()
        
                if self._update_scheduler == 'es':
                    self._lr_scheduler.step()
            
            prev_lr = get_lr()
            logger.info(f'Current learning rate: {prev_lr}')
            
            cum_loss /= (nb + 1)
            logger.info(f'Train loss: {cum_loss}')

            dec_metric = 0.
            if self._val_dataset is not None:
                _, __, val_metrics = self._model.predict(self._val_dataset, evaluate=True, 
                                                         metrics=self._validation_metrics)
                val_loss = val_metrics[0]
                logger.info(f'Validation loss: {val_loss}')
                logger.info(f'Validation metrics: {val_metrics[1:]}')
                
                dec_metric = self._decision_metric(val_metrics)
                
                if self._keep_best_model and (dec_metric < best_dec_metric):
                    best_model = copy.deepcopy(self._model._bert_model.state_dict())
                    best_dec_metric = dec_metric
            
            if self._update_scheduler == 'ee':
                self._lr_scheduler.step(dec_metric)
                
            if self._restore_bm_on_lr_change and get_lr() < prev_lr:
                if get_lr() < self._smallest_lr: 
                    iterator.close()
                    break

                prev_lr = get_lr()
                logger.info(f'Reduced learning rate to: {prev_lr}')
                    
                logger.info('Restoring best model...')
                self._model._bert_model.load_state_dict(best_model)

        if best_model:
            self._model._bert_model.load_state_dict(best_model)

        torch.cuda.empty_cache()
