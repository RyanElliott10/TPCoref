import torch
import torchtext.data as data


def my_tokenize(s):
    return s.split('\n')


# TEXT = data.Field(sequential=True, init_token='<sod>', eos_token='<eod>', tokenize=my_tokenize)
# LABEL = data.Field(sequential=True, init_token='<sod>', eos_token='<eod>', dtype=torch.long)
TEXT = data.Field(sequential=True, tokenize=my_tokenize)
LABEL = data.Field(sequential=True, dtype=torch.long)

fields = [('passage', TEXT), ('references', LABEL)]
refs = data.TabularDataset(path='../data/processed/dev.tsv', format='tsv',
                           fields=fields, skip_header=True)

TEXT.build_vocab(refs)
LABEL.build_vocab(refs)

data_iter = data.BucketIterator(
    refs, batch_size=100, sort_key=lambda x: len(x.passage), shuffle=True)

for e in data_iter:
    print(e.passage)
    print(e.references)

    print(list(map(lambda x: TEXT.vocab.itos[x], e.passage[:, 0].tolist())))
    print(list(map(lambda x: TEXT.vocab.itos[x], e.references[:, 0].tolist())))
    break
