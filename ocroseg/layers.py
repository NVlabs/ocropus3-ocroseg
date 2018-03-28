import sys

import numpy as np
import torch
from torch import nn
from torch.legacy import nn as legnn
from torch.autograd import Variable

sys.modules["layers"] = sys.modules["ocroseg.layers"]

BD = "BD"
LBD = "LBD"
LDB = "LDB"
BDL = "BDL"
BLD = "BLD"
BWHD = "BWHD"
BDWH = "BDWH"
BWH = "BWH"


def lbd2bdl(x):
    assert len(x.size()) == 3
    return x.permute(1, 2, 0).contiguous()


def bdl2lbd(x):
    assert len(x.size()) == 3
    return x.permute(2, 0, 1).contiguous()


def typeas(x, y):
    """Make x the same type as y, for numpy, torch, torch.cuda."""
    assert not isinstance(x, Variable)
    if isinstance(y, Variable):
        y = y.data
    if isinstance(y, np.ndarray):
        return asnd(x)
    if isinstance(x, np.ndarray):
        if isinstance(y, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = torch.FloatTensor(x)
        else:
            x = torch.DoubleTensor(x)
    return x.type_as(y)


class Viewer(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        return "Viewer %s" % (self.shape, )


class Textline2Img(nn.Module):
    iorder = BWH
    oorder = BDWH

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, seq):
        b, l, d = seq.size()
        return seq.view(b, 1, l, d)


class Img2Seq(nn.Module):
    iorder = BDWH
    oorder = BDL

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, img):
        b, d, w, h = img.size()
        perm = img.permute(0, 1, 3, 2).contiguous()
        return perm.view(b, d * h, w)


class ImgMaxSeq(nn.Module):
    iorder = BDWH
    oorder = BDL

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, img):
        # BDWH -> BDW -> BWD
        return img.max(3)[0].squeeze(3)


class ImgSumSeq(nn.Module):
    iorder = BDWH
    oorder = BDL

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, img):
        # BDWH -> BDW -> BWD
        return img.sum(3)[0].squeeze(3).permute(0, 2, 1).contiguous()


class Lstm1(nn.Module):
    """A simple bidirectional LSTM."""
    iorder = BDL
    oorder = BDL

    def __init__(self, ninput=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        assert ninput is not None
        assert noutput is not None
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=self.ndir - 1)

    def forward(self, seq, volatile=False):
        seq = bdl2lbd(seq)
        l, bs, d = seq.size()
        assert d == self.ninput, seq.size()
        h0 = Variable(typeas(torch.zeros(self.ndir, bs, self.noutput), seq), volatile=volatile)
        c0 = Variable(typeas(torch.zeros(self.ndir, bs, self.noutput), seq), volatile=volatile)
        post_lstm, _ = self.lstm(seq, (h0, c0))
        return lbd2bdl(post_lstm)


class Lstm2to1(nn.Module):
    """An LSTM that summarizes one dimension."""
    iorder = BDWH
    oorder = BDL

    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, img, volatile=False):
        # BDWH -> HBWD -> HBsD
        b, d, w, h = img.size()
        seq = img.permute(3, 0, 2, 1).contiguous().view(h, b * w, d)
        bs = b * w
        h0 = Variable(typeas(torch.zeros(1, bs, self.noutput), img), volatile=volatile)
        c0 = Variable(typeas(torch.zeros(1, bs, self.noutput), img), volatile=volatile)
        # HBsD -> HBsD
        assert seq.size() == (h, b * w, d), (seq.size(), (h, b * w, d))
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm.size() == (h, b * w, self.noutput), (post_lstm.size(), (h, b * w, self.noutput))
        # HBsD -> BsD -> BWD
        final = post_lstm.select(0, h - 1).view(b, w, self.noutput)
        assert final.size() == (b, w, self.noutput), (final.size(), (b, w, self.noutput))
        # BWD -> BDW
        final = final.permute(0, 2, 1).contiguous()
        assert final.size() == (b, self.noutput, w), (final.size(), (b, self.noutput, self.noutput))
        return final


class Lstm1to0(nn.Module):
    """An LSTM that summarizes one dimension."""
    iorder = BDL
    oorder = BD

    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, seq):
        volatile = not isinstance(seq, Variable) or seq.volatile
        seq = bdl2lbd(seq)
        l, b, d = seq.size()
        assert d == self.ninput, (d, self.ninput)
        h0 = Variable(typeas(torch.zeros(1, b, self.noutput), seq), volatile=volatile)
        c0 = Variable(typeas(torch.zeros(1, b, self.noutput), seq), volatile=volatile)
        assert seq.size() == (l, b, d)
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm.size() == (l, b, self.noutput)
        final = post_lstm.select(0, l - 1).view(b, self.noutput)
        return final


class RowwiseLSTM(nn.Module):
    def __init__(self, ninput=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=self.ndir - 1)

    def forward(self, img):
        volatile = not isinstance(img, Variable) or img.volatile
        b, d, h, w = img.size()
        # BDHW -> WHBD -> WB'D
        seq = img.permute(3, 2, 0, 1).contiguous().view(w, h * b, d)
        # WB'D
        h0 = typeas(torch.zeros(self.ndir, h * b, self.noutput), img)
        c0 = typeas(torch.zeros(self.ndir, h * b, self.noutput), img)
        h0 = Variable(h0, volatile=volatile)
        c0 = Variable(c0, volatile=volatile)
        seqresult, _ = self.lstm(seq, (h0, c0))
        # WB'D' -> BD'HW
        result = seqresult.view(w, h, b, self.noutput * self.ndir).permute(2, 3, 1, 0)
        return result


class Lstm2(nn.Module):
    """A 2D LSTM module."""

    def __init__(self, ninput=None, noutput=None, nhidden=None, ndir=2):
        nn.Module.__init__(self)
        assert ndir in [1, 2]
        nhidden = nhidden or noutput
        self.hlstm = RowwiseLSTM(ninput, nhidden, ndir=ndir)
        self.vlstm = RowwiseLSTM(nhidden * ndir, noutput, ndir=ndir)

    def forward(self, img):
        horiz = self.hlstm(img)
        horizT = horiz.permute(0, 1, 3, 2).contiguous()
        vert = self.vlstm(horizT)
        vertT = vert.permute(0, 1, 3, 2).contiguous()
        return vertT
