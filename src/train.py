import argparse
import time
from typing import List

import spacy
import torch
import torch.nn as nn

from dataset import get_dataiter
from model import Transformer, PyTransformer

torch.manual_seed(42)
torch.set_printoptions(threshold=50000)

nlp = spacy.load('en')

D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.
LR = 1e-3
N_EPOCHS = 5
B_SIZE = 20


def train(train_iter, val_iter, TEXT, LABEL):
    global D_MODEL, N_LAYERS, N_HEADS, DROPOUT, N_EPOCHS, LR
    SRC_V_SIZE = len(TEXT.vocab)
    TGT_V_SIZE = len(LABEL.vocab)

    model = Transformer(SRC_V_SIZE, TGT_V_SIZE, D_MODEL, N_LAYERS, N_HEADS, dropout=DROPOUT)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)

    criterion = nn.CrossEntropyLoss()
    logsoftmax = nn.LogSoftmax(dim=-1)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print(f'Model Hyperparameters\n---------------------\nModel Hidden Dimension: {D_MODEL}'
          f'\nNum Layers: {N_LAYERS}\nNum Attention Heads: {N_HEADS}\nDropout: {DROPOUT}'
          f'\nLearning Rate: {LR}\nNum Epochs: {N_EPOCHS}\nBatch Size: {B_SIZE}'
          f'\nSource Vocab Size: {SRC_V_SIZE}\nTarget Vocab Size: {TGT_V_SIZE}\n')

    for epoch in range(N_EPOCHS):
        running_loss = 0.
        for b_num, batch in enumerate(train_iter):
            model.train()
            start_time = time.time()

            src_input = batch.passage
            tgt_input = batch.references

            # FOR DEBUG
            # src_input = torch.tensor([[0, 0], [43, 78], [23, 23], [89, 45],
            #                           [34, 1], [10, 2], [29, 2], [1, 2]])
            # tgt_input = torch.tensor([[0, 0], [9, 9], [4, 3], [6, 7],
            #                           [9, 1], [9, 2], [6, 2], [1, 2]])

            src = src_input.transpose(0, 1)
            tgt = tgt_input.transpose(0, 1)

            if epoch == b_num == 0:
                if args.verbose:
                    print(f'src_input:\t{src_input.shape}\n{src_input}')
                    print(f'tgt_input:\t{tgt_input.shape}\n{tgt_input}')
                else:
                    print(f'src_input:\t{src_input.shape}')
                    print(f'tgt_input:\t{tgt_input.shape}')

            SRC_LEN = src.size(-1)
            TGT_LEN = tgt.size(-1)

            src_mask = None
            tgt_mask = Transformer.generate_square_subsequent_mask(TGT_LEN)
            src_key_padding_mask = (src == TEXT.vocab.stoi['<pad>'])
            tgt_key_padding_mask = (tgt == LABEL.vocab.stoi['<pad>'])
            memory_key_padding_mask = (src == TEXT.vocab.stoi['<pad>'])

            if epoch == b_num == 0:
                if args.verbose:
                    print(f'src_mask:\t{src_mask}')
                    print(f'tgt_mask:\t{tgt_mask.shape}\n{tgt_mask}')
                    print(f'src_key_padding_mask:\t{src_key_padding_mask.shape}'
                          f'\n{src_key_padding_mask}')
                    print(f'tgt_key_padding_mask:\t{tgt_key_padding_mask.shape}'
                          f'\n{tgt_key_padding_mask}')
                    print(f'memory_key_padding_mask:\t{memory_key_padding_mask.shape}'
                          f'\n{memory_key_padding_mask}')
                else:
                    print(f'src_mask:\t{src_mask}')
                    print(f'tgt_mask:\t{tgt_mask.shape}')
                    print(f'src_key_padding_mask:\t{src_key_padding_mask.shape}')
                    print(f'tgt_key_padding_mask:\t{tgt_key_padding_mask.shape}')
                    print(f'memory_key_padding_mask:\t{memory_key_padding_mask.shape}')

            output = model(src_input, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            if epoch == b_num == 0:
                if args.verbose:
                    print(f'output:\t{output.shape}\n{output}')
                    print(f'output (view):\t{output.view(-1, TGT_V_SIZE).shape}'
                          f'\n{output.view(-1, TGT_V_SIZE)}')
                    print(f'tgt:\t{tgt.shape}\n{tgt}')
                    print(f'tgt (view):\t{tgt.contiguous().view(-1).shape}'
                          f'\n{tgt.contiguous().view(-1)}')
                else:
                    print(f'output:\t\t{output.shape}')
                    print(f'output (view):\t{output.view(-1, TGT_V_SIZE).shape}')
                    print(f'tgt:\t\t{tgt.contiguous().shape}')
                    print(f'tgt (view):\t{tgt.contiguous().view(-1).shape}\n')

            loss = criterion(output.view(-1, TGT_V_SIZE), tgt.contiguous().view(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()
            # scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            running_loss += loss
            el_time = time.time() - start_time

            if b_num == 0:
                print(f'Epoch {epoch+1}/{N_EPOCHS}')
            print(f'\tBatch {b_num}/{len(train_iter)} | secs/batch: {round(el_time, 4)}'
                  f' | loss: {loss} | lr: {scheduler.get_last_lr()}')

            if b_num % 10 == 0 and b_num != 0:
                val_iter = evaluate(model, val_iter, TEXT, LABEL)

        print(f'Epoch {epoch}/{N_EPOCHS} | loss: {running_loss}')
        torch.save(model.state_dict(), f'../models/train/train{epoch}.pth')

    torch.save(model.state_dict(), f'../models/train/goldtrain.pth')
    src = "This is a random sentence lol I wonder how this works."
    # predict(model, src, TEXT, LABEL, custom_sent=True)


def evaluate(model, val_iter, TEXT, LABEL):
    """Returns an iterator to let it test multiple validation datapoints rather than strictly the
    first.
    """
    TGT_V_SIZE = len(LABEL.vocab)

    model.eval()
    total_loss = 0.

    criterion = nn.CrossEntropyLoss()
    logsoftmax = nn.LogSoftmax(dim=-1)

    with torch.no_grad():
        for batch in val_iter:
            src_input = batch.passage
            tgt_input = batch.references
            src = src_input.transpose(0, 1)
            tgt = tgt_input.transpose(0, 1)

            SRC_LEN = src.size(-1)
            TGT_LEN = tgt.size(-1)

            src_mask = None
            tgt_mask = Transformer.generate_square_subsequent_mask(TGT_LEN)
            src_key_padding_mask = (src == TEXT.vocab.stoi['<pad>'])
            tgt_key_padding_mask = (tgt == LABEL.vocab.stoi['<pad>'])
            memory_key_padding_mask = (src == TEXT.vocab.stoi['<pad>'])

            output = model(src_input, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            total_loss += criterion(output.view(-1, TGT_V_SIZE), tgt.contiguous().view(-1))
            break
    print(f'\tValidation loss: {total_loss}')

    # DO NOT DELETE. THIS SHOWS HOW TO GET THE ACTUAL VALUES
    print(f'argmax:\n{logsoftmax(output).argmax(dim=-1).view(-1, TGT_LEN)}')
    return val_iter


def add_sp_tokens(src: List[str], TEXT) -> torch.Tensor:
    src = [tok.text for tok in nlp(src)]
    src.insert(0, '<sod>')
    src.append('<eod>')
    src_t = torch.tensor([TEXT.vocab.stoi[t] for t in src])

    print(src_t.shape)
    print(src_t)

    return src_t


def predict(model, src, TEXT, LABEL, custom_sent: bool = False):
    model.eval()

    if custom_sent:
        add_sp_tokens(src, TEXT)

    TEXT_pad_tok = TEXT.vocab.stoi['<pad>']
    LABEL_pad_tok = LABEL.vocab.stoi['<pad>']

    src_pad_mask = (src == TEXT_pad_tok).view(1, src.size(0))
    src.unsqueeze_(1)

    e_out = model.encoder(src, src_pad_mask)

    outputs = torch.zeros(src.size(0)).type_as(src.data)
    outputs[0] = LABEL.vocab.stoi['<sod>']

    # I'm not certain this is right... look at
    # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f
    for _ in range(1, outputs.size(0)):
        tgt_mask = Transformer.generate_square_subsequent_mask(outputs.size(0))
        tgt_pad_mask = (outputs == LABEL_pad_tok).view(1, outputs.size(0))
        out = model.decoder(outputs.unsqueeze(1), e_out, tgt_mask=tgt_mask,
                            tgt_pad_mask=tgt_pad_mask)

    preds = out.argmax(2)
    for el in preds:
        print(el.item())
        print(LABEL.vocab.itos[el.item()])


def main(args):
    global D_MODEL, N_LAYERS, N_HEADS, DROPOUT, N_EPOCHS, B_SIZE, LR

    D_MODEL = args.modeldim
    N_LAYERS = args.nlayers
    N_HEADS = args.nheads
    DROPOUT = args.dropout
    N_EPOCHS = args.epochs
    B_SIZE = args.batchsize
    LR = args.learningrate

    train_iter, val_iter, TEXT, LABEL = get_dataiter(args.datapath, batch_size=B_SIZE)

    if args.predict:
        model = Transformer(len(TEXT.vocab), len(LABEL.vocab), D_MODEL,
                            N_LAYERS, N_HEADS, dropout=DROPOUT)
        model = torch.load('../models/ahem.pth')
        predict(model, args.predicts, TEXT, LABEL, custom_sent=True)
        exit(0)

    train(train_iter, val_iter, TEXT, LABEL)


def safe_debug():
    """Important takeaways: when viewing the model's output: the line should look like:
        >>> logsoftmax(output).argmax(dim=-1).view(-1, TGT_LEN)

    Further, src_mask should (generally) be none.
    Keep src and tgt in their (BATCH_SIZE, SRC/TGT_LEN) shape, and transpose into src/tgt_input:
        >>> src_input = src.transpose(0, 1)
        >>> tgt_input = tgt.trasnpose(0, 1)

    SRC/TGT_LEN variables should be found in each batch.
        >>> SRC_LEN = src.size(-1)
        >>> TGT_LEN = tgt.size(-1)

    Masks use the raw src/tgt inputs with (BATCH_SIZE, SRC/TGT_LEN) shape
        >>> src_mask = None
        >>> tgt_mask = Transformer.generate_square_subsequent_mask(TGT_LEN)
        >>> src_key_padding_mask = (src == 2)
        >>> tgt_key_padding_mask = (tgt == 2)
        >>> memory_key_padding_mask = (src == 2)

    I believe <sod>, <eod>, and <pad> tokens improve learning.

    A learning rate too high tends to go to a uniform single number output.
    """
    BATCH_SIZE = 2
    EPOCHS = 50
    BATCHES = 15
    SRC_V_SIZE = 12938
    TGT_V_SIZE = 845

    model = Transformer(SRC_V_SIZE, TGT_V_SIZE, 512, 4, 8, 0.)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    logsoftmax = nn.LogSoftmax(dim=-1)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for epoch in range(EPOCHS):
        model.train()
        for batch in range(BATCHES):
            src_input = torch.tensor([[0, 0], [43, 78], [23, 23], [89, 45], [
                                     34, 1], [10, 2], [29, 2], [1, 2]])
            tgt_input = torch.tensor([[0, 0], [9, 9], [4, 3], [6, 7],
                                      [9, 1], [9, 2], [6, 2], [1, 2]])

            src = src_input.transpose(0, 1)
            tgt = tgt_input.transpose(0, 1)

            if epoch == batch == 0:
                if args.verbose:
                    print(f'src_input:\t{src_input.shape}\n{src_input}')
                    print(f'tgt_input:\t{tgt_input.shape}\n{tgt_input}')
                else:
                    print(f'src_input:\t{src_input.shape}')
                    print(f'tgt_input:\t{tgt_input.shape}')

            SRC_LEN = src.size(-1)
            TGT_LEN = tgt.size(-1)

            src_mask = None
            tgt_mask = Transformer.generate_square_subsequent_mask(TGT_LEN)
            src_key_padding_mask = (src == 2)
            tgt_key_padding_mask = (tgt == 2)
            memory_key_padding_mask = (src == 2)

            if epoch == batch == 0:
                if args.verbose:
                    print(f'src_mask:\t{src_mask}')
                    print(f'tgt_mask:\t{tgt_mask.shape}\n{tgt_mask}')
                    print(f'src_key_padding_mask:\t{src_key_padding_mask.shape}'
                          f'\n{src_key_padding_mask}')
                    print(f'tgt_key_padding_mask:\t{tgt_key_padding_mask.shape}'
                          f'\n{tgt_key_padding_mask}')
                    print(f'memory_key_padding_mask:\t{memory_key_padding_mask.shape}'
                          f'\n{memory_key_padding_mask}')
                else:
                    print(f'src_mask:\t{src_mask}')
                    print(f'tgt_mask:\t{tgt_mask.shape}')
                    print(f'src_key_padding_mask:\t{src_key_padding_mask.shape}')
                    print(f'tgt_key_padding_mask:\t{tgt_key_padding_mask.shape}')
                    print(f'memory_key_padding_mask:\t{memory_key_padding_mask.shape}')

            output = model(src_input, tgt_input, src_mask, tgt_mask, src_key_padding_mask,
                           tgt_key_padding_mask, memory_key_padding_mask)

            if epoch == batch == 0:
                if args.verbose:
                    print(f'output:\t{output.shape}\n{output}')
                    print(f'output (view):\t{output.view(-1, TGT_V_SIZE).shape}'
                          f'\n{output.view(-1, TGT_V_SIZE)}')
                    print(f'tgt:\t{tgt.shape}\n{tgt}')
                    print(f'tgt (view):\t{tgt.contiguous().view(-1).shape}'
                          f'\n{tgt.contiguous().view(-1)}')
                else:
                    print(f'output:\t\t{output.shape}')
                    print(f'output (view):\t{output.view(-1, TGT_V_SIZE).shape}')
                    print(f'tgt:\t\t{tgt.contiguous().shape}')
                    print(f'tgt (view):\t{tgt.contiguous().view(-1).shape}\n')

            loss = criterion(output.view(-1, TGT_V_SIZE), tgt.contiguous().view(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f'Epoch {epoch}/{EPOCHS} loss: {loss}')
        print(f'output:\n\t{output.shape}\n{output}')
        print(f'tgt:\n{tgt}')
        print(f'argmax:\n{logsoftmax(output).argmax(dim=-1).view(-1, TGT_LEN)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--datapath', required=True, help='Filepath to the dataset tsv file')
    parser.add_argument('-md', '--modeldim', type=int, default=D_MODEL, help='d_model param')
    parser.add_argument('-nl', '--nlayers', type=int, default=N_LAYERS, help='n_layers param')
    parser.add_argument('-nh', '--nheads', type=int, default=N_HEADS, help='n_heads param')
    parser.add_argument('-do', '--dropout', type=float, default=DROPOUT, help='dropout param')
    parser.add_argument('-lr', '--learningrate', type=float, default=LR,
                        help='Learning Rate')
    parser.add_argument('-e', '--epochs', type=int, default=N_EPOCHS, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=B_SIZE, help='Batch size')
    parser.add_argument('-p', '--predict', type=str, default=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
    # safe_debug()
