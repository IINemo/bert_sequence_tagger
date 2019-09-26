import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_transformers.optimization import WarmupLinearSchedule

from pytorch_transformers import AdamW

import copy
from tqdm import trange

import logging
logger = logging.getLogger('sequence_tagger_bert')


from fastai.basic_train import Learner
from fastai.callback import OptimWrapper
from fastai.basic_data import DataBunch
from fastai.torch_core import flatten_model



def ner_loss_func(out, ys): 
    '''
    Loss function - to use with fastai learner
    It calculates the loss for token classification using softmax cross entropy
    If out is already the loss, we simply return the loss
    '''
    
    # If out is already the loss
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        

    labels = ys
    active_loss = labels != 0
    active_logits = out.view(-1, out.shape[-1])[active_loss]
    active_labels = labels.view(-1)[active_loss]
    loss = loss_fct(active_logits, active_labels)
    loss = loss.mean(-1)
    return loss


class FastAiTrainerBert:
    def __init__(self, model, optimizer, train_dataloader, val_dataloader, epoch, lr):
        self._model = model
        self._optimizer = optimizer
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._epoch = epoch
        self._lr = lr
        
    def train(self):
        fastai_optimizer = OptimWrapper(self._optimizer)
        
        data_bunch = DataBunch(
            train_dl=self._train_dataloader,
            valid_dl=self._val_dataloader
        )
        
        learner = Learner(data_bunch, self._model, BertAdam,
                    loss_func=ner_loss_func,
                    #metrics=metrics,
                    true_wd=False,
                    layer_groups=None,
                    path='learn')
        learner.optim = fastai_optimizer
        
        for epoch in range(self._epoch):
            learner.fit(1, self._lr)


class ModelTrainerBert:
    def __init__(self, 
                 model, 
                 optimizer, 
                 train_dataloader, 
                 val_dataloader, 
                 anneal_factor=0.5,
                 patience=1,
                 reduce_on_plateau=True,
                 number_of_steps=-1,
                 warmup_proportion=0.1):
        self._model = model
        self._optimizer = optimizer
        self._reduce_on_plateau = reduce_on_plateau
#         if self._reduce_on_plateau:
#             self._lr_scheduler = ReduceLROnPlateau(self._optimizer, 
#                                                    factor=anneal_factor, 
#                                                    patience=patience, 
#                                                    verbose=True, 
#                                                    mode='min')
#         else:
#             self._lr_scheduler = WarmupLinearSchedule(self._optimizer, 
#                                                        t_total=number_of_steps, 
#                                                        warmup_steps=int(warmup_proportion*number_of_steps))
            
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
    
    def train(self, epochs, max_grad_norm=1.0, max_to_anneal=2):
        n_anneal = 0
        best_model = {}
        
        get_lr = lambda: self._optimizer.param_groups[0]['lr']
        prev_lr = get_lr()
        
        iterator = trange(epochs, desc="Epoch")
        for epoch in iterator:
            self._model._bert_model.train()

            cum_loss = 0.
            for tokens, labels in self._train_dataloader:
                #self._lr_scheduler.step()
                
                #loss, mask_sum = self._model.forward_loss(tokens, labels)
                loss = self._model.forward_loss(tokens, labels)
                loss.backward()
                
                #cum_loss += (loss.item() / mask_sum)
                cum_loss += loss.item()
#                 torch.nn.utils.clip_grad_norm_(parameters=self._model._bert_model.parameters(), 
#                                                max_norm=max_grad_norm)
                self._optimizer.step()
                self._model._bert_model.zero_grad()

            logger.info(f'Train loss: {cum_loss}')

            if self._val_dataloader is not None:
                _, val_loss, val_f1 = self._model.predict(self._val_dataloader, evaluate=True)
                logger.info(f'Validation loss: {val_loss}')
                logger.info(f'Validation F1-Score: {val_f1}')
                
                if not self._reduce_on_plateau:
                    continue
                    
                if val_loss < self._lr_scheduler.best:
                    best_model = copy.deepcopy(self._model._bert_model.state_dict())
                
                self._lr_scheduler.step(val_loss)
                
                if get_lr() < prev_lr:
                    n_anneal += 1
                
                    if n_anneal > max_to_anneal:
                        iterator.close()
                        break

                    prev_lr = get_lr()
                    
                    logger.info('Restoring best model...')
                    self._model._bert_model.load_state_dict(best_model)

        if best_model:
            self._model._bert_model.load_state_dict(best_model)

        torch.cuda.empty_cache()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.optimizer import required


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

def initBertAdam(params, lr, warmup=0.1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), e=1e-6, weight_decay=0.01, max_grad_norm=1.0):
    return BertAdam(params, lr, warmup, t_total, schedule, betas, e, weight_decay, max_grad_norm)

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), e=1e-6, weight_decay=0.01, max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid betas[0] parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid betas[1] parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)

        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['betas']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss        






def create_optimizer(model, full_finetuning, t_total, lr_body, weight_decay):
    return initBertAdam(model.parameters(), lr=lr_body, weight_decay=weight_decay, t_total=t_total)
    #return AdamW(model.parameters(), lr=lr_body, correct_bias=False, weight_decay=weight_decay)

        
# def create_optimizer(model, full_finetuning, lr_body=5e-5, 
#                      lr_head=5e-4, weight_decay=0.1):
#     no_decay = ['bias', 'gamma', 'beta']

#     logger.info(f'Full finetuning: {full_finetuning}')
#     if full_finetuning:
#         param_optimizer = list(model.bert.named_parameters())
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer 
#                         if not any(nd in n for nd in no_decay)],
#              'weight_decay': weight_decay},
#             {'params': [p for n, p in param_optimizer 
#                         if any(nd in n for nd in no_decay)],
#              'weight_decay': 0.0},
#             {'params' : [p for n, p in model.classifier.named_parameters()
#                          if not any(nd in n for nd in no_decay)],
#              'lr' : lr_head,
#              'weight_decay': weight_decay},
#             {'params' : [p for n, p in model.classifier.named_parameters()
#                         if any(nd in n for nd in no_decay)],
#              'lr' : lr_head,
#              'weight_decay' : 0.0
#             }
#         ]
        
#         n_params = sum(p.numel() for p in model.parameters())
#     else:
#         param_optimizer = list(model.classifier.named_parameters()) 
#         optimizer_grouped_parameters = [
#             {'params' : [p for n, p in param_optimizer
#                          if not any(nd in n for nd in no_decay)],
#              'lr' : lr_head,
#              'weight_decay': weight_decay},
#             {'params' : [p for n, p in param_optimizer
#                          if any(nd in n for nd in no_decay)],
#              'lr' : lr_head,
#              'weight_decay' : 0.0
#             }
#         ]
    
#         n_params = sum(p.numel() for p in model.classifier.parameters())
        
#     logger.info(f'N parameters: {n_params}')

#     optimizer = AdamW(optimizer_grouped_parameters, lr=lr_body, correct_bias=False)
    
#     return optimizer
