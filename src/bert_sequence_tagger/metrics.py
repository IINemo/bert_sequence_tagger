import itertools
from sklearn.metrics import f1_score as f1_score_sklearn
from seqeval.metrics import f1_score


def f1_entity_level(*args, **kwargs):
    return f1_score(*args, **kwargs)


def f1_token_level(true_labels, predictions):
    true_labels = list(itertools.chain(*true_labels))
    predictions = list(itertools.chain(*predictions))
    
    labels = list(set(true_labels) - {'[PAD]', 'O'})
    
    return f1_score_sklearn(true_labels, 
                            predictions, 
                            average='micro',
                            labels=labels)
