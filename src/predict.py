import argparse
from typing import List

import spacy
import torch
import torch.nn.functional as F

from data import get_dataiter
from model import Transformer
from train import create_src_masks, create_tgt_masks

nlp = spacy.load('en')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

D_MODEL = 768
N_LAYERS = 4
N_HEADS = 12
DROPOUT = 0.2
B_SIZE = 1

TEXT = None
LABEL = None


def add_sp_tokens(src: List[str]) -> torch.Tensor:
    global TEXT

    src = [tok.text for tok in nlp(src)]
    src.insert(0, '<sod>')
    src.append('<eod>')
    return torch.tensor([TEXT.vocab.stoi[t] for t in src]).view(-1, 1)


def predict(model, src):
    global TEXT, LABEL

    src_input = add_sp_tokens(src)[1:, :].to(device)  # Cut off first <sod> token on src
    tgt_input = torch.zeros(src_input.shape).long().to(device)

    tgt_input[0, :] = LABEL.vocab.stoi['<sod>']

    row_src = src_input.transpose(0, 1).to(device)
    row_tgt = tgt_input.transpose(0, 1).to(device)

    SRC_SEQ_LEN = row_src.size(-1)
    TGT_SEQ_LEN = row_tgt.size(-1)

    assert SRC_SEQ_LEN == src_input.size(0) and TGT_SEQ_LEN == tgt_input.size(0)

    src_mask, src_key_padding_mask, memory_key_padding_mask = create_src_masks(
        row_src, SRC_SEQ_LEN, TEXT, use_srcmask=False)
    encoded = model.encoder(src=src_input, src_mask=src_mask,
                            src_key_padding_mask=src_key_padding_mask)

    for i in range(1, SRC_SEQ_LEN):
        tgt_mask, tgt_key_padding_mask = create_tgt_masks(row_tgt, i, LABEL)
        tgt_key_padding_mask = tgt_key_padding_mask[:, :i]

        decoded = model.decoder(tgt=tgt_input[:i], memory=encoded,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        proj = model.linear(decoded)
        preds = F.log_softmax(proj, dim=-1).argmax(dim=-1)
        tgt_input[i] = preds[-1]

    return tgt_input


def main():
    global TEXT, LABEL

    _, _, TEXT, LABEL = get_dataiter(args.datapath, batch_size=B_SIZE)
    SRC_V_SIZE = len(TEXT.vocab)
    TGT_V_SIZE = len(LABEL.vocab)

    model = Transformer(SRC_V_SIZE, TGT_V_SIZE, D_MODEL, N_LAYERS,
                        N_HEADS, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(args.modelpath, map_location=device))
    model.eval()

    preds = predict(model, args.document)
    print(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, required=True, help='Filepath to load model')
    parser.add_argument('--datapath', type=str, required=True, help='Path to data files')
    parser.add_argument('--document', type=str, required=True, help='Document to predict on')
    parser.add_argument('--modeldim', type=int, default=D_MODEL, help='d_model param')
    parser.add_argument('--nlayers', type=int, default=N_LAYERS, help='n_layers param')
    parser.add_argument('--nheads', type=int, default=N_HEADS, help='n_heads param')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='dropout param')
    args = parser.parse_args()

    main()
