import math

import torch
import torch.nn as nn

Tensor = torch.Tensor
BoolTensor = torch.BoolTensor
ByteTensor = torch.ByteTensor


class PositionalEncoder(nn.Module):
    """Positional encoding class pulled from the PyTorch documentation tutorial
    on Transformers for seq2seq models:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self, v_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.):
        """

        """
        super(TransformerEncoder, self).__init__()

        self.v_size = v_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.embedding = nn.Embedding(self.v_size, self.d_model)
        self.pe = PositionalEncoder(self.d_model, dropout=self.dropout, max_len=2000)
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                           dropout=self.dropout)
        self.t_encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=self.n_layers)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: ByteTensor, src_key_padding_mask: BoolTensor):
        """Forward pass on the Transformer Encoder.

        :param src: Inputs to be encoded.
        :param pad_mask: Mask for inputs to hide padding.

        Shape:
            - src: :math:`(source sequence length, batch size)`
            - src_key_padding_mask: :math:`(batch size, source sequence length)`

            where feature number is the 1-hot index corresponding to the token.
            Further, note that these shapes are perceived as different from the
            PyTorch source code, but the additional src dimension (feature num)
            is added after `src` passes through the embedding layer, therefore
            adding the third dimension (acting as word embeddings).

        Example:
            >>> VOCAB_SIZE = 10000
            >>> SRC_LEN = 50
            >>> BATCH_SIZE = 100
            >>> src = torch.randint(0, VOCAB_SIZE, (SRC_LEN, BATCH_SIZE))
            >>> pad_mask = (src < 7500).bool().view(BATCH_SIZE, SRC_LEN)
            >>> encoder = Encoder(VOCAB_SIZE, d_model=512, n_layers=6, n_heads=4)
            >>> encoder(src, pad_mask)
        """
        embeds = self.embedding(src) * math.sqrt(self.d_model)
        positions = self.pe(embeds)
        encoded = self.t_encoder(positions, mask=src_mask,
                                 src_key_padding_mask=src_key_padding_mask)
        return encoded


class TransformerDecoder(nn.Module):

    def __init__(self, v_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.):
        """

        """
        super(TransformerDecoder, self).__init__()

        self.v_size = v_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.embedding = nn.Embedding(self.v_size, self.d_model)
        self.pe = PositionalEncoder(d_model=self.d_model, dropout=self.dropout, max_len=2000)
        layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                           dropout=self.dropout)
        self.t_decoder = nn.TransformerDecoder(decoder_layer=layer, num_layers=self.n_layers)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: ByteTensor,
        tgt_key_padding_mask: BoolTensor,
        memory_key_padding_mask: BoolTensor
    ):
        """Forward pass on the Transformer Decoder.

        :param tgt: Target output (necessary for training).
        :param memory: Output from the Encoder layer.
        :param tgt_mask: Lower triangular matrix to ensure the decoder doesn't
                         look at future tokens in a given subsequence.
        :param tgt_pad_mask: Mask padding in the target sequence.
        :param mem_pad_mask: Mask memory padding in the target sequence
                             (generally the same as tgt_pad_mask)
        positioons = self.pe(tgt)
        decoded = self.t_decoder(tgt=tgt, memory=memory,
                                 tgt_key_padding_mask=tgt_mask,
                                 memory_key_padding_mask=mem_mask)

        Shapes:
            - tgt: :math:`(target sequence length, batch size)`
            - memory: :math:`(source sequece length, batch size, feature number)`
            - tgt_mask: :math:`(target sequence length, target sequence length)`
            - tgt_pad_mask: :math:`(batch size, target sequence length)`
            - mem_pad_mask: :math:`(batch size, memory sequence length)`

        Example:
            >>>
        """
        embeds = self.embedding(tgt) * math.sqrt(self.d_model)
        positions = self.pe(embeds)
        decoded = self.t_decoder(tgt=positions, memory=memory, tgt_mask=tgt_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
        return decoded


class Transformer(nn.Module):

    def __init__(
        self,
        src_v_size: int,
        label_v_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.
    ):
        super(Transformer, self).__init__()

        self.src_v_size = src_v_size
        self.label_v_size = label_v_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.encoder = TransformerEncoder(self.src_v_size, self.d_model, self.n_layers,
                                          self.n_heads, dropout=self.dropout)
        self.decoder = TransformerDecoder(self.label_v_size, self.d_model, self.n_layers,
                                          self.n_heads, dropout=self.dropout)
        self.linear = nn.Linear(self.d_model, self.label_v_size)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: ByteTensor,
        tgt_mask: ByteTensor,
        src_key_padding_mask: BoolTensor,
        tgt_key_padding_mask: BoolTensor,
        memory_key_padding_mask: BoolTensor
    ):
        encoded = self.encoder(src=src, src_mask=src_mask,
                               src_key_padding_mask=src_key_padding_mask)
        decoded = self.decoder(tgt=tgt, memory=encoded, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        proj = self.linear(decoded)
        return proj

    @staticmethod
    def generate_square_subsequent_mask(seq_size: int):
        """Generate a square mask for the sequence. Returns a byte tensor (rather than a float
        tensor) to mask the indexes (per the PyTorch source code, this is allowed) to avoid the
        issue with softmax and float('-inf').
        """
        mask = torch.triu(torch.ones(seq_size, seq_size), diagonal=1).byte()
        return mask


class TPEncoder(nn.Module):
    pass


class TPDecoder(nn.Module):
    pass


class TransformerPointer(nn.Module):
    pass


class PyTransformer(nn.Module):

    def __init__(
        self,
        src_v_size: int,
        label_v_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.
    ):
        super(PyTransformer, self).__init__()

        self.embedding = nn.Embedding(src_v_size, d_model)
        self.transformer = nn.Transformer(d_model, n_heads, n_layers, n_layers)
        self.linear = nn.Linear(512, label_v_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        src_key_padding_mask: BoolTensor,
        tgt_key_padding_mask: BoolTensor,
        memory_key_padding_mask: BoolTensor = None
    ):
        src_embeds = self.embedding(src)
        tgt_embeds = self.embedding(tgt)
        output = self.transformer(src=src_embeds, tgt=tgt_embeds, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        output = self.linear(output)
        return output
