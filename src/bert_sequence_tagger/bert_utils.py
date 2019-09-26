
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
