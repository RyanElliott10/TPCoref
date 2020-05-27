import argparse
import time
from typing import List

import spacy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataiter
from model import Transformer

torch.manual_seed(42)
torch.set_printoptions(threshold=50000)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nlp = spacy.load('en')

D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
DROPOUT = 0.2
LR = 1e-4
N_EPOCHS = 10
B_SIZE = 4


def train(train_iter, val_iter, TEXT, LABEL):
    """After extensive testing, I've found a few errors in how we are evaluating the predictions.
    Firstly, we should always use LogSoftmax(dim=-1) and argmax(dim=-1) when evaluating. But
    Furthermore, CrossEntropyLoss needs to accept a tensor alternating examples. So, we should use
    tgt.view(-1) where tgt.shape = (TGT_SEQ_LEN, BATCH_SIZE).
    """
    global D_MODEL, N_LAYERS, N_HEADS, DROPOUT, N_EPOCHS, LR
    SRC_V_SIZE = len(TEXT.vocab)
    TGT_V_SIZE = len(LABEL.vocab)

    model = Transformer(SRC_V_SIZE, TGT_V_SIZE, D_MODEL, N_LAYERS,
                        N_HEADS, dropout=DROPOUT).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.95)

    criterion = nn.CrossEntropyLoss()

    print(f'Model Hyperparameters\n---------------------\nModel Hidden Dimension: {D_MODEL}'
          f'\nNum Layers: {N_LAYERS}\nNum Attention Heads: {N_HEADS}\nDropout: {DROPOUT}'
          f'\nLearning Rate: {LR}\nNum Epochs: {N_EPOCHS}\nBatch Size: {B_SIZE}'
          f'\nSource Vocab Size: {SRC_V_SIZE}\nTarget Vocab Size: {TGT_V_SIZE}\n')

    loss_interval = 128
    loss_values_sum = 0.
    loss_values = []
    validation_values = []

    model.train()
    for epoch in range(N_EPOCHS):
        running_loss = 0.
        print(f'Epoch {epoch+1}/{N_EPOCHS}')

        for b_num, batch in enumerate(train_iter):
            torch.cuda.empty_cache()
            start_time = time.time()
            true_batch_num = (len(train_iter) * epoch) + b_num

            src_input = batch.passage.to(device)
            tgt_input = batch.references.to(device)

            src = src_input.transpose(0, 1).to(device)
            tgt = tgt_input.transpose(0, 1).to(device)

            if epoch == b_num == 0:
                if args.verbose:
                    print(f'src_input:\t{src_input.shape}\n{src_input}')
                    print(f'tgt_input:\t{tgt_input.shape}\n{tgt_input}')
                else:
                    print(f'src_input:\t{src_input.shape}')
                    print(f'tgt_input:\t{tgt_input.shape}')

            SRC_SEQ_LEN = src.size(-1)
            TGT_SEQ_LEN = tgt.size(-1)

            src_mask = None
            tgt_mask = Transformer.generate_square_subsequent_mask(TGT_SEQ_LEN).to(device)
            src_key_padding_mask = (src == TEXT.vocab.stoi['<pad>']).to(device)
            tgt_key_padding_mask = (tgt == LABEL.vocab.stoi['<pad>']).to(device)
            memory_key_padding_mask = (src == TEXT.vocab.stoi['<pad>']).to(device)

            if epoch == b_num == 0:
                if args.verbose:
                    print(f'src_mask:\t{src_mask.shape if src_mask is not None else src_mask}')
                    print(f'tgt_mask:\t{tgt_mask.shape}\n{tgt_mask}')
                    print(f'src_key_padding_mask:\t{src_key_padding_mask.shape}'
                          f'\n{src_key_padding_mask}')
                    print(f'tgt_key_padding_mask:\t{tgt_key_padding_mask.shape}'
                          f'\n{tgt_key_padding_mask}')
                    print(f'memory_key_padding_mask:\t{memory_key_padding_mask.shape}'
                          f'\n{memory_key_padding_mask}')
                else:
                    print(f'src_mask:\t{src_mask.shape if src_mask is not None else src_mask}')
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
                    print(f'tgt_input:\t{tgt_input.shape}\n{tgt_input}')
                    print(f'tgt_input (view):\t{tgt_input.view(-1).shape}'
                          f'\n{tgt_input.view(-1)}')
                else:
                    print(f'output:\t\t{output.shape}')
                    print(f'output (view):\t{output.view(-1, TGT_V_SIZE).shape}')
                    print(f'tgt_input:\t\t{tgt_input.shape}')
                    print(f'tgt_input (view):\t{tgt_input.view(-1).shape}\n')

            loss = criterion(output.view(-1, TGT_V_SIZE), tgt_input.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_values_sum += loss.item()
            running_loss += loss.item()
            el_time = time.time() - start_time

            if b_num % loss_interval == 0 and b_num > 0:
                loss_values.append((true_batch_num, loss_values_sum / loss_interval))
                loss_values_sum = 0.

            if b_num % 128 == 0:
                print(f'\tBatch {b_num}/{len(train_iter)} | secs/batch: '
                      f'{round(el_time, 4)} | loss: {loss} | '
                      f'lr: {scheduler.get_last_lr()}')

            if b_num % (len(train_iter) // 10) == 0 and b_num > 0:
                val_loss = evaluate(model, val_iter, TEXT, LABEL)
                model.train()

                validation_values.append((true_batch_num, val_loss))
                plt.plot(*zip(*loss_values), label='Train Loss')
                plt.plot(*zip(*validation_values), label='Validation Loss')
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

        scheduler.step()
        print(f'Epoch {epoch+1}/{N_EPOCHS} | loss: {running_loss}')
        if epoch != N_EPOCHS-1:
            save_path = f'../models/train/train{epoch}.pth'
            torch.save(model.state_dict(), save_path)

    print(f'Expected output shape {tgt_input.shape}\nTargets:{tgt_input}')
    print(f'output raw shape: {output.shape}\nargmax:\n{format_preds(output, TGT_SEQ_LEN)}')

    save_path = '../models/train/goldtrain.pth'
    torch.save(model.state_dict(), save_path)


def format_preds(output, TGT_SEQ_LEN):
    """Returns a tensor of shape (TGT_SEQ_LEN, BATCH_SIZE). So to iterate over a single example, you go
    over the tensor row by row.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    return log_softmax(output).argmax(dim=-1).transpose(0, 1)


def evaluate(model, val_iter, TEXT, LABEL):
    """Returns an iterator to let it test multiple validation datapoints rather than strictly the
    first.
    """
    TGT_V_SIZE = len(LABEL.vocab)

    model.eval()
    total_loss = 0.

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_iter:
            src_input = batch.passage.to(device)
            tgt_input = batch.references.to(device)
            src = src_input.transpose(0, 1).to(device)
            tgt = tgt_input.transpose(0, 1).to(device)

            SRC_SEQ_LEN = src.size(-1)
            TGT_SEQ_LEN = tgt.size(-1)

            src_mask = None
            tgt_mask = Transformer.generate_square_subsequent_mask(TGT_SEQ_LEN).to(device)
            src_key_padding_mask = (src == TEXT.vocab.stoi['<pad>']).to(device)
            tgt_key_padding_mask = (tgt == LABEL.vocab.stoi['<pad>']).to(device)
            memory_key_padding_mask = (src == TEXT.vocab.stoi['<pad>']).to(device)

            output = model(src_input, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            total_loss += criterion(output.view(-1, TGT_V_SIZE), tgt_input.view(-1)).item()
    print(f'\tValidation loss: {total_loss}')
    print(f'Expected:\n{tgt[0, :]}')
    print(f'argmax:\n{format_preds(output, TGT_SEQ_LEN)[0, :]}')
    return total_loss / len(val_iter)


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

    print('src:', src)

    model = Transformer(len(TEXT.vocab), len(LABEL.vocab), 512, 6, 8)

    # src_pad_mask = (src == TEXT_pad_tok).view(1, src.size(0))
    # src.unsqueeze_(1)

    # e_out = model.encoder(src, src_pad_mask)

    # outputs = torch.zeros(src.size(0)).type_as(src.data)
    # outputs[0] = LABEL.vocab.stoi['<sod>']

    # # I'm not certain this is right... look at
    # # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f
    # for _ in range(1, outputs.size(0)):
    #     tgt_mask = Transformer.generate_square_subsequent_mask(outputs.size(0))
    #     tgt_pad_mask = (outputs == LABEL_pad_tok).view(1, outputs.size(0))
    #     out = model.decoder(outputs.unsqueeze(1), e_out, tgt_mask=tgt_mask,
    #                         tgt_pad_mask=tgt_pad_mask)

    # preds = out.argmax(2)
    # for el in preds:
    #     print(el.item())
    #     print(LABEL.vocab.itos[el.item()])


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
        model = torch.load('../models/train/5epochs/goldtrain.pth',
                           map_location=torch.device('cpu'))
        predict(model, args.predicts, TEXT, LABEL, custom_sent=True)
        exit(0)

    train(train_iter, val_iter, TEXT, LABEL)


def safe_debug():
    """Important takeaways: when viewing the model's output: the line should look like:
        >>> log_softmax(output).argmax(dim=-1).view(-1, TGT_SEQ_LEN)

    Further, src_mask should (generally) be none.
    Keep src and tgt in their (BATCH_SIZE, SRC/TGT_SEQ_LEN) shape, and transpose into src/tgt_input:
        >>> src_input = src.transpose(0, 1)
        >>> tgt_input = tgt.trasnpose(0, 1)

    SRC/TGT_SEQ_LEN variables should be found in each batch.
        >>> SRC_SEQ_LEN = src.size(-1)
        >>> TGT_SEQ_LEN = tgt.size(-1)

    Masks use the raw src/tgt inputs with (BATCH_SIZE, SRC/TGT_SEQ_LEN) shape
        >>> src_mask = None
        >>> tgt_mask = Transformer.generate_square_subsequent_mask(TGT_SEQ_LEN)
        >>> src_key_padding_mask = (src == 2)
        >>> tgt_key_padding_mask = (tgt == 2)
        >>> memory_key_padding_mask = (src == 2)

    I believe <sod>, <eod>, and <pad> tokens improve learning.

    A learning rate too high tends to go to a uniform single number output.

    I'm no longer certain this is correct.
    """
    BATCH_SIZE = 2
    EPOCHS = 50
    BATCHES = 15
    SRC_V_SIZE = 12938
    TGT_V_SIZE = 845

    model = Transformer(SRC_V_SIZE, TGT_V_SIZE, 512, 4, 8, 0.).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for batch in range(BATCHES):
            src_input = torch.tensor([[0, 0], [43, 78], [23, 23], [89, 45],
                                      [34, 1], [10, 2], [29, 2], [1, 2]]).to(device)
            tgt_input = torch.tensor([[0, 0], [9, 9], [4, 3], [6, 7],
                                      [9, 1], [9, 2], [6, 2], [1, 2]]).to(device)

            src = src_input.transpose(0, 1).to(device)
            tgt = tgt_input.transpose(0, 1).to(device)

            if epoch == batch == 0:
                if args.verbose:
                    print(f'src_input:\t{src_input.shape}\n{src_input}')
                    print(f'tgt_input:\t{tgt_input.shape}\n{tgt_input}')
                else:
                    print(f'src_input:\t{src_input.shape}')
                    print(f'tgt_input:\t{tgt_input.shape}')

            SRC_SEQ_LEN = src.size(-1)
            TGT_SEQ_LEN = tgt.size(-1)

            src_mask = Transformer.generate_square_subsequent_mask(SRC_SEQ_LEN).to(device)
            tgt_mask = Transformer.generate_square_subsequent_mask(TGT_SEQ_LEN).to(device)
            src_key_padding_mask = (src == 2).to(device)
            tgt_key_padding_mask = (tgt == 2).to(device)
            memory_key_padding_mask = (src == 2).to(device)

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
                    print(f'tgt_input:\t{tgt_input.shape}\n{tgt_input}')
                    print(f'tgt_input (view):\t{tgt_input.view(-1).shape}'
                          f'\n{tgt_input.view(-1)}')
                else:
                    print(f'output:\t\t{output.shape}')
                    print(f'output (view):\t{output.view(-1, TGT_V_SIZE).shape}')
                    print(f'tgt_input:\t\t{tgt_input.shape}')
                    print(f'tgt_input (view):\t{tgt_input.view(-1).shape}\n')

            loss = criterion(output.view(-1, TGT_V_SIZE), tgt_input.view(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f'Epoch {epoch}/{EPOCHS} loss: {loss}')
        print(f'tgt:\n{tgt}')
        print(f'argmax:\n{format_preds(output, TGT_SEQ_LEN)[0, :]}')


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
