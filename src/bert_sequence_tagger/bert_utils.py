from torch.utils.data import DataLoader


def make_bert_tag_dict_from_flair_corpus(corpus):
    tags_vals = corpus.make_tag_dictionary('ner').get_items()
    tags_vals.remove('<unk>')
    tags_vals.remove('<START>')
    tags_vals.remove('<STOP>')
    tags_vals = ['[PAD]'] + tags_vals # + ['X']#, '[CLS]', '[SEP]']
    tag2idx = {t : i for i, t in enumerate(tags_vals)}
    return tags_vals, tag2idx


def prepare_flair_corpus(corpus, name='ner', filter_tokens={'-DOCSTART-'}):
    result = []
    for sent in corpus:
        if sent[0].text in filter_tokens:
            continue
        else:
            result.append(([token.text for token in sent.tokens],
                           [token.tags[name].value for token in sent.tokens]))
    
    return result


def get_parameters_without_decay(model, no_decay={'bias', 'gamma', 'beta'}):
    params_no_decay = []
    params_decay = []
    for n, p in model.named_parameters():
        if any((e in n) for e in no_decay):
            params_no_decay.append(p)
        else:
            params_decay.append(p)
    
    return [{'params' : params_no_decay, 'weight_decay' : 0.},
            {'params' : params_decay}]


def get_model_parameters(model, no_decay={'bias', 'gamma', 'beta'}, 
                         full_finetuning=True, lr_head=None):
    grouped_parameters = get_parameters_without_decay(model.classifier, no_decay)
    if lr_head is not None:
        for param in grouped_parameters:
            param['lr'] = lr_head
    
    if full_finetuning:
        grouped_parameters = (get_parameters_without_decay(model.bert, no_decay) 
                              + grouped_parameters)
    
    return grouped_parameters


def create_loader_from_flair_corpus(corpus, sampler_ctor, batch_size):
    collate_fn = lambda inpt: tuple(zip(*inpt))
    
    dataset = prepare_flair_corpus(corpus)
    sampler = sampler_ctor(dataset)
    dataloader = DataLoader(dataset, 
                            sampler=sampler, 
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    return dataloader
    