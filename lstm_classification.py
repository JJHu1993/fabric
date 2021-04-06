import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn

class LstmClassification(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, bidirectional = False):
        super(LstmClassification, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = 2
        self.drop = nn.Dropout(p = 0.1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = self.num_layers, batch_first = True, bidirectional = bidirectional)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.atten = SoftDotAttention(hidden_dim*self.num_directions)

    def forward(self, inputs, lengths): #  dim = batch*seq_len*input_dim
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first = True)
        ctx,(hidden,_) = self.lstm(packed_inputs)
        # ctx, hidden = self.lstm(inputs)

        ctx, _ = pad_packed_sequence(ctx, batch_first=True)
        hidden = self.drop(hidden)
        ctx = self.drop(ctx)

        # hidden_cat = hidden[0]
        # for i in range(1,len(hidden)):
        #     hidden_cat  = torch.cat((hidden_cat, hidden[i]), 1)
        if self.num_directions == 2:
            hidden = torch.cat((hidden[-1], hidden[-2]), 1)
        else:
            hidden = hidden[-1]

        mask = np.array([[True for _ in range(lengths[0])] for _ in range(len(lengths))])
        for i in range(len(lengths)):
            mask[i,:lengths[i]] = [False for _ in range(lengths[i])]
        mask = torch.from_numpy(mask).bool().cuda()
        hidden_tile, _ = self.atten(hidden, ctx, mask)
        output = self.linear(hidden_tile)

        return output



