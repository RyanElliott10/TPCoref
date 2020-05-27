import argparse
import os
import sys
import time
from collections import defaultdict
from typing import DefaultDict, List, Tuple

import pandas as pd
import torch
import torchtext.data as data

PRECO_BASEDIR = os.getenv('PRECO_DATADIR')
NO_REF_TOK = '<nr>'
BLACK_QUOTES = ['”', "''", "``", "“"]
SPLIT_CHAR = '\n'  # there are spaces in the dataset so we gotta use \n. Baffling, really.

debug = False

FlattenedIndicesType = List[Tuple[int, int]]
TokenMapType = DefaultDict[int, List[int]]

"""A big issue with the current parsing of this dataset is that it chooses the most recent entity
when an entity is a child of some other entity. For example:

    When Jerry ran to the park with me ... my dad bought me a car.

the phrase "my dad" contains two entities, and is tagged as follows:

    entity 1: "my"
    entity 2: "my dad"

The person who wrote this parser (I refuse to accept blame) MAY tag "my" as the same entity as
["my dad", "Jerry"] and NOT ["me", "me"].

The solution to this lies in the fact that the leading word is typically a modifier, meaning it
shouldn't be tagged as the "larger" entity. But this selection process requires some state
indicating that you're within a mult-tagged entity parse.
"""


def get_dataiter(filepath: str, batch_size: int = 100):
    """Creates a dataiter object for the dataset provided at the filepath.

    :param filepath: String of raw filepath to dataset.tsv

    Example:
        get_dataiter('../data/processed/dev.tsv')
    """
    def my_tokenize(s):
        return s.split(SPLIT_CHAR)

    TEXT = data.Field(sequential=True, init_token='<sod>', eos_token='<eod>', tokenize=my_tokenize)
    LABEL = data.Field(sequential=True, init_token='<sod>', eos_token='<eod>', dtype=torch.long)

    fields = [('passage', TEXT), ('references', LABEL)]

    train, val = data.TabularDataset.splits(path=filepath, train='train.tsv', validation='dev.tsv',
                                            format='tsv', fields=fields, skip_header=True)
    train_iter, val_iter = data.BucketIterator.splits((train, val),
                                                      batch_sizes=(batch_size, batch_size),
                                                      sort_key=lambda x: len(x.passage),
                                                      shuffle=True)

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    return train_iter, val_iter, TEXT, LABEL


class Entity(object):

    def __init__(self):
        self.idxs = list()
        self.proposed = None

    def set_proposed_idx(self, idx: int):
        """Acts as a state, is "cleared" upon each call (since it's initially None)."""
        proposed = None
        proposed_diff = sys.maxsize
        for el in self.idxs:
            use_idx = el[1]
            diff = idx - use_idx
            if diff >= 0:  # if the use_idx is less than the index we're looking for
                # update with least distance between (last reference)
                proposed_diff = min(proposed_diff, diff)
                proposed = el if proposed_diff == diff else proposed
        self.proposed = proposed


def clean_quotes(s): return '"' if s in BLACK_QUOTES else s  # a quirk in the dataset


def clean_tok(s):
    s = clean_quotes(s)
    return s.lower()


def flatten(l): return [clean_quotes(item) for sublist in l for item in sublist]


def adjust_indices(
    sents,
    idxs
) -> Tuple[List[str], FlattenedIndicesType, TokenMapType, List[Entity]]:
    """The idea is to flatten all passages into a single 1D list. To do this, we need to adjust all
    the indices. This function flattens a given passage and transforms the indices to match.

    Returns:
        flat_sents: 1D list containing all tokens in the passage.
        new_idxs: Adjusted indices to match flat_sents.
        tok_map: Dictionary with keys as raw indices into the flattened passage. Each key contains
                 a list of entities that refer to it. To be used in creating the mapping from token
                 to referent.
        ents: List of entities (aligned with tok_map)
    """
    lens = list(map(lambda x: len(x), sents))
    flat_sents = flatten(sents)
    new_idxs = []
    tok_map = defaultdict(list)
    ents = list()

    for ent_num, ent_idxs in enumerate(idxs):
        new_idxs.append([])
        ent = Entity()
        for idx in ent_idxs:
            base_offset = sum(lens[0:idx[0]])
            start = idx[1] + base_offset
            end = idx[2] + base_offset
            adjusted_idxs = (start, end)
            ent.idxs.append(adjusted_idxs)
            new_idxs[-1].append(adjusted_idxs)
            for i in range(start, end):
                tok_map[i].append(ent_num)
        ents.append(ent)
    return flat_sents, new_idxs, tok_map, ents


def get_ref_token(
    idx: int,
    proposed_ent_idxs: List[int],
    ents: List[Entity],
    no_ref: str
) -> Tuple[int, int]:
    """Gets the reference token for a given entity.

    :param idx: The index of the entity we're looking at
    :param proposed_ent_idxs: A list of entity indices that may be considered
    :param ents: A list of entities, accessed via proposed_ent_idxs[i]
    :param no_ref: if there aren't references, use this token. Generally, the length of the passage.
    """
    if len(proposed_ent_idxs) == 0:
        return (no_ref, no_ref)[-1]
    [ents[ent_idx].set_proposed_idx(idx) for ent_idx in proposed_ent_idxs]
    proposed = (idx, idx+1)  # fallback, point to self if first reference
    for idx in proposed_ent_idxs:
        # This is where the issue mentioned above lies. This is a naive algorithm that just assigns
        # the last entity (index-wise) to the entity we're trying to tag.
        proposed = ents[idx].proposed if ents[idx].proposed is not None else proposed

    proposed = (proposed[0], proposed[1])
    return proposed[-1]


def parse_referents(
    sents: List[str],
    idxs: FlattenedIndicesType,
    tok_map: TokenMapType,
    ents: List[Entity]
):
    """We must now create a new list, for each index, and point to the previous reference for that
    word.
    """
    references = list()
    for i in range(len(sents)):
        # ref_idx = get_ref_token(i, tok_map[i], ents, len(sents))
        ref_idx = get_ref_token(i, tok_map[i], ents, NO_REF_TOK)
        references.append(ref_idx)
    return references


def main():
    global debug
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-f', '--file', required=True,
                        help="The PreCo data file to be parsed. One of dev.json or train.json")
    parser.add_argument('-s', '--savepath',
                        help="The filepath to save the csv file with parsed data.")
    parser.add_argument('-t', '--time', help='Option to show estimated time remaining',
                        action='store_true')
    parser.add_argument('-r', '--read', action='store_true')
    args = parser.parse_args()
    debug = args.debug

    if args.read:
        import numpy as np
        data = pd.read_csv(f'{args.file}', sep='\t')
        np_arr = data.to_numpy().astype('object')
        print(np_arr.shape)
        print(np.asarray(data.loc[0, 'tokenized_passage']))
        exit(0)

    global_data = pd.DataFrame(columns=['tokenized_passage', 'references'])
    data = pd.read_json(f'{PRECO_BASEDIR}/{args.file}', lines=True)

    start_time = time.time()

    for idx, row in data.iterrows():
        trunc = row.to_numpy()
        sents = trunc[1]
        idxs = trunc[2]
        sents, idxs, tok_map, ents = adjust_indices(sents, idxs)
        references = parse_referents(sents, idxs, tok_map, ents)

        assert len(sents) == len(references)

        if debug:
            for i, ref in enumerate(references):
                if ref[0] == NO_REF_TOK:
                    print(f'{sents[i]}: {ref}')
                else:
                    print(f'{sents[i]}: {ref} {sents[ref[0]:ref[1]+1]}')
        df = pd.DataFrame()
        df['tokenized_passage'] = [SPLIT_CHAR.join(sents)]
        df['references'] = [SPLIT_CHAR.join(str(x) for x in references)]
        global_data = global_data.append(df, ignore_index=True)

        if idx % 500 == 0:
            tper_row = ((time.time() - start_time) / 500)
            rows_remaining = data.shape[0] - idx
            start_time = time.time()
            if args.time:
                print(f'Progress: {idx} / {data.shape[0]}\tEstimated Time Remaining: '
                      f'{round((tper_row * rows_remaining)/60, 4)} minutes')
            else:
                print(f'Progress: {idx} / {data.shape[0]}')

    global_data.to_csv(args.savepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
