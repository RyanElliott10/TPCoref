import argparse
import sys
import time
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_dataiter
from model import Transformer

IN_COLAB = 'google.colab' in sys.modules

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_printoptions(threshold=50000)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nlp = spacy.load('en')

D_MODEL = 0
N_LAYERS = 0
N_HEADS = 0
DROPOUT = 0.
LR = 0.
N_EPOCHS = 0
B_SIZE = 0


def create_src_masks(src, SRC_SEQ_LEN, TEXT):
    src_mask = None
    src_key_padding_mask = (src == TEXT.vocab.stoi['<pad>']).bool().to(device)
    memory_key_padding_mask = (src == TEXT.vocab.stoi['<pad>']).bool().to(device)
    return src_mask, src_key_padding_mask, memory_key_padding_mask


def create_tgt_masks(tgt, TGT_SEQ_LEN, LABEL):
    tgt_mask = Transformer.generate_square_subsequent_mask(TGT_SEQ_LEN).to(device)
    tgt_key_padding_mask = (tgt == LABEL.vocab.stoi['<pad>']).bool().to(device)
    return tgt_mask, tgt_key_padding_mask


def ed_train(train_iter, val_iter, TEXT, LABEL):
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
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.90)

    criterion = nn.CrossEntropyLoss()

    print(f'Encoder/Decoder Model Hyperparameters\n---------------------\nModel Hidden Dimension: {D_MODEL}'
          f'\nNum Layers: {N_LAYERS}\nNum Attention Heads: {N_HEADS}\nDropout: {DROPOUT}'
          f'\nLearning Rate: {LR}\nNum Epochs: {N_EPOCHS}\nBatch Size: {B_SIZE}'
          f'\nSource Vocab Size: {SRC_V_SIZE}\nTarget Vocab Size: {TGT_V_SIZE}\n')

    loss_interval = 128
    loss_values = []
    val_loss_values = []
    val_acc_values = []

    model.train()
    for epoch in range(1, N_EPOCHS+1):
        running_loss = 0.
        loss_values_sum = 0.
        print(f'Epoch {epoch}/{N_EPOCHS}')

        for b_num, batch in enumerate(train_iter):
            torch.cuda.empty_cache()
            start_time = time.time()
            true_batch_num = (len(train_iter) * epoch-1) + b_num

            # The actual target (that we're using to calculate loss) is expecting words from index
            # through n. Think of it this way: if we're trying to generate the sequence of words:
            # ["<sod>", "He", "ran", "to", "the", "park", "<eod>"]
            # Then we want to compare the output against ["He", "ran", "to", "the", "park", "<eod>"],
            # with the input taking the form of          ["<sod>", "He", "ran", "to", "the", "park"],
            # where
            # ["<sod>"] generates "He",
            # ["<sod>", "He"], generates "ran"
            # etc.
            #
            # Therefore, the inputs need to be "shifted" to the left, and the outputs need to be
            # "shifted" to the right. This prevents the Transformer from copying the tgt_input to
            # the outputs
            src_input, tgt_input, row_src, row_tgt = parse_batch(batch)

            SRC_SEQ_LEN = row_src.size(-1)
            TGT_SEQ_LEN = row_tgt.size(-1)

            src_mask, src_key_padding_mask, memory_key_padding_mask = create_src_masks(
                row_src, SRC_SEQ_LEN, TEXT)
            tgt_mask, tgt_key_padding_mask = create_tgt_masks(row_tgt, TGT_SEQ_LEN, LABEL)

            output = model(src_input, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            loss = criterion(output.view(-1, TGT_V_SIZE), row_tgt.contiguous().view(-1))
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

            if b_num % (len(train_iter) // 5) == 0 and b_num > 0:
                val_loss, val_acc = ed_evaluate(model, val_iter, TEXT, LABEL)
                model.train()

                val_loss_values.append((true_batch_num, val_loss))
                val_acc_values.append((true_batch_num, val_acc))

                if len(val_loss_values) > 1:
                    plt.plot(*zip(*loss_values), label='Train Loss')
                    plt.plot(*zip(*val_loss_values), label='Validation Loss')
                    plt.xlabel('Batch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()

                if len(val_acc_values) > 1:
                    plt.plot(*zip(*val_acc_values), label='Validation Accuracy')
                    plt.xlabel('Batch')
                    plt.ylabel('Accuracy')
                    plt.ylim(0, 1)
                    plt.legend()
                    plt.show()

        scheduler.step()
        print(f'Epoch {epoch}/{N_EPOCHS} | loss: {running_loss}')
        if epoch != N_EPOCHS:
            save_path = f'{args.savepath}train{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            try:
                files.download(save_path)
            except:
                print(f'Unable to download {save_path}')

    print(f'Expected output shape {tgt_input.shape}\nTargets:{tgt_input}')
    print(f'output raw shape: {output.shape}\nargmax:\n{format_preds(output, TGT_SEQ_LEN)}')

    save_path = f'{args.savepath}goldtrain.pth'
    torch.save(model.state_dict(), save_path)


def ed_evaluate(model, val_iter, TEXT, LABEL):
    TGT_V_SIZE = len(LABEL.vocab)

    model.eval()
    total_loss = 0.
    criterion = nn.CrossEntropyLoss()

    n_correct = 0
    total_vals = 0

    with torch.no_grad():
        for batch in val_iter:
            src_input, tgt_input, row_src, row_tgt = parse_batch(batch)

            SRC_SEQ_LEN = row_src.size(-1)
            TGT_SEQ_LEN = row_tgt.size(-1)

            src_mask, src_key_padding_mask, memory_key_padding_mask = create_src_masks(
                row_src, SRC_SEQ_LEN, TEXT)
            tgt_mask, tgt_key_padding_mask = create_tgt_masks(row_tgt, TGT_SEQ_LEN, LABEL)

            output = model(src_input, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            expected = row_tgt.contiguous().view(-1)
            total_loss += criterion(output.view(-1, TGT_V_SIZE), expected).item()

            argmaxed = F.log_softmax(output.view(-1, TGT_V_SIZE), dim=-1).argmax(dim=-1)
            n_correct += (argmaxed == expected).byte().sum().item()
            total_vals += expected.size(0)

    avg_loss = total_loss / len(val_iter)
    acc = n_correct / total_vals
    print(f'Expected:\n{row_tgt[0, :]}')
    print(f'argmax:\n{format_preds(output, TGT_SEQ_LEN)[0, :]}')
    print(f'\tValidation loss: {avg_loss}\n\tValidation accuracy: {acc}')

    return avg_loss, acc


def parse_batch(batch):
    src_input = batch.passage.to(device)
    tgt_input = batch.references[:-1, :].to(device)
    row_src = batch.passage.transpose(0, 1)
    row_tgt = batch.references[1:, :].transpose(0, 1).to(device)
    return src_input, tgt_input, row_src, row_tgt


def format_preds(output, TGT_SEQ_LEN):
    """Return a tensor of shape (TGT_SEQ_LEN, BATCH_SIZE)."""
    return F.log_softmax(output, dim=-1).argmax(dim=-1).transpose(0, 1)


def add_sp_tokens(src: List[str], TEXT) -> torch.Tensor:
    src = [tok.text for tok in nlp(src)]
    src.insert(0, '<sod>')
    src.append('<eod>')
    return torch.tensor([TEXT.vocab.stoi[t] for t in src])


def predict(model, src, TEXT, LABEL, custom_sent: bool = False):
    pass


def main():
    global D_MODEL, N_LAYERS, N_HEADS, DROPOUT, N_EPOCHS, B_SIZE, LR

    D_MODEL = args.modeldim
    N_LAYERS = args.nlayers
    N_HEADS = args.nheads
    DROPOUT = args.dropout
    N_EPOCHS = args.epochs
    B_SIZE = args.batchsize
    LR = args.lr

    train_iter, val_iter, TEXT, LABEL = get_dataiter(args.datapath, batch_size=B_SIZE)

    if args.predict:
        model = Transformer(len(TEXT.vocab), len(LABEL.vocab), D_MODEL,
                            N_LAYERS, N_HEADS, dropout=DROPOUT)
        model = torch.load(args.predmodel, map_location=torch.device('cpu'))
        predict(model, args.predict, TEXT, LABEL, custom_sent=True)
        exit(0)

    print(f'Training start time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    if args.linear:
        el_train(train_iter, val_iter, TEXT, LABEL)
    else:
        ed_train(train_iter, val_iter, TEXT, LABEL)
    print(f'Training completion time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


if __name__ == '__main__':
    if IN_COLAB:
        class ColabArgs(object):
            def __init__(self):
                self.datapath = './'
                self.savepath = './models/'
                self.modeldim = 512
                self.nlayers = 6
                self.nheads = 8
                self.dropout = 0.2
                self.lr = 1e-3
                self.epochs = 15
                self.batchsize = 8
                self.predict = False
                self.linear = False
                self.verbose = False
                self.predmodel = '../models/train/encoder_decoder/noref_tok/10epochs/goldtrain.pth'

        args = ColabArgs()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--datapath', required=True, help='Filepath to the dataset tsv file')
        parser.add_argument('--savepath', required=True, help='Filepath save .pth files')
        parser.add_argument('--modeldim', type=int, default=D_MODEL, help='d_model param')
        parser.add_argument('--nlayers', type=int, default=N_LAYERS, help='n_layers param')
        parser.add_argument('--nheads', type=int, default=N_HEADS, help='n_heads param')
        parser.add_argument('--dropout', type=float, default=DROPOUT, help='dropout param')
        parser.add_argument('--lr', type=float, default=LR,
                            help='Learning Rate')
        parser.add_argument('--epochs', type=int, default=N_EPOCHS, help='Number of epochs')
        parser.add_argument('--batchsize', type=int, default=B_SIZE, help='Batch size')
        parser.add_argument('--predict', type=str, default=False)
        parser.add_argument('--predmodel', type=str)
        parser.add_argument('--linear', action='store_true')
        parser.add_argument('-v', '--verbose', action='store_true')
        args = parser.parse_args()

    main()
