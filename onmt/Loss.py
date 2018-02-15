"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt


class LossComputeBase(nn.Module):
    """
    This is the loss criterion base class. Users can implement their own
    loss computation strategy by making subclass of this one.
    Users need to implement the compute_loss() and make_shard_state() methods.
    We inherits from nn.Module to leverage the cuda behavior.
    """
    def __init__(self, generator, tgt_vocab, tgt_vocab2):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.tgt_vocab2 = tgt_vocab2
        self.padding_idx = tgt_vocab.stoi[onmt.IO.PAD_WORD]
        self.padding_idx2 = tgt_vocab2.stoi[onmt.IO.PAD_WORD]

    def make_shard_state(self, batch, output, labels, labels_free, target, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns, labels, labels_free, target, labels_add):
        """
        Compute the loss monolithically, not dividing into shards.
        """
        range_ = (0, batch.tgt.size(0))
        #shard_state = self.make_shard_state(batch, output, labels, labels_free, target, range_, attns)
        #_, batch_stats, _, _ = self.compute_loss(batch, **shard_state)

        _, batch_stats, _, _= self.compute_loss(batch, output, batch.tgt[range_[0]+1:range_[1]], labels, labels_free, target, labels_add)
        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             labels, labels_free, target, lamb, labels_add):
        """
        Compute the loss in shards for efficiency.
        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        #shard_state = self.make_shard_state(batch, output, labels, labels_free, target, range_, attns, labels_add)

        lr_factor = 5
        #for shard in shards(shard_state, shard_size):
        #loss, stats, loss2, loss3 = self.compute_loss(batch, **shard)
        loss, stats, loss2, loss3 = self.compute_loss(batch, output, batch.tgt[range_[0]+1:range_[1]], labels, labels_free, target, labels_add)
        if self.phase == 1:
            loss1 = - loss3*lamb + loss
            for p in self.encoder.classifier.parameters():
                p.requires_grad = False
            if self.encoder.num_classifiers > 1:
                for i in range(self.encoder.num_classifiers-1):
                    c = getattr(self.encoder, 'classifier%d'%i)
                    for p in c.parameters():
                        p.requires_grad = False
            loss1.div(batch.batch_size).backward(retain_graph=True)
            for p in self.encoder.classifier.parameters():
                p.requires_grad = True
            if self.encoder.num_classifiers > 1:
                for i in range(self.encoder.num_classifiers-1):
                    c = getattr(self.encoder, 'classifier%d'%i)
                    for p in c.parameters():
                        p.requires_grad = True
            loss2.div(float(batch.batch_size)/self.encoder.num_classifiers/lr_factor).backward()
        elif self.phase == 2:
            #print ('Warning: NMT loss ignored')
            loss2.div(float(batch.batch_size)).backward()
        else:
            assert False
        batch_stats.update(stats)

        return batch_stats

    def stats(self, loss, scores, target):
        """
        Compute and return a Statistics object.

        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def stats_all(self, loss, scores, target, loss2, scores2, target2):
        """
        Compute and return a Statistics object.

        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        pred2 = scores2.max(1)[1]
        non_padding2 = target2.ne(self.padding_idx2)
        num_correct2 = pred2.eq(target2) \
                          .masked_select(non_padding2) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct, loss2[0], non_padding2.sum(), num_correct2)

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, tgt_vocab2, encoder, phase):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab, tgt_vocab2)

        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)
        self.phase = phase
        weight2 = torch.ones(len(tgt_vocab2))
        weight2[self.padding_idx2] = 0
        self.criterion2 = nn.CrossEntropyLoss(weight2, size_average=False)
        self.criterion3 = nn.CrossEntropyLoss(weight2, size_average=False)
        self.criterions2 = []
        self.criterions3 = []
        self.encoder = encoder
        if self.encoder.num_classifiers>1:
            print ('Using %d classifiers'%self.encoder.num_classifiers)
            for i in range(self.encoder.num_classifiers-1):
                self.criterions2.append(nn.CrossEntropyLoss(weight2, size_average=False))
                self.criterions3.append(nn.CrossEntropyLoss(weight2, size_average=False))

    def make_shard_state(self, batch, output, labels, labels_free, target, range_, attns=None):
        """ See base class for args description. """
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "labels": labels,
            "labels_free": labels_free,
            "target2": target,
        }

    def compute_loss(self, batch, output, target, labels, labels_free, target2, labels_add):
        """ See base class for args description. """
        scores = self.generator(self.bottle(output))

        target = target.view(-1)

        loss = self.criterion(scores, target)
        loss_data = loss.data.clone()

        #print ('labels')
        #print (labels.size())
        #print ('target2')
        #print (target2.contiguous().size())
        labels = labels.view(-1, labels.size(2))
        labels_free = labels_free.view(-1, labels_free.size(2))
        target2 = target2.contiguous().view(-1)
        loss2 = self.criterion2(labels, target2)
        loss_data2 = loss2.data.clone()
        loss3 = self.criterion3(labels_free, target2)
        for i in range(self.encoder.num_classifiers-1):
            loss2 = loss2 + self.criterions2[i](labels_add[i][0].view(-1, labels_free.size(1)), target2)
            loss3 = loss3 + self.criterions3[i](labels_add[i][1].view(-1, labels_free.size(1)), target2)
        loss2 = loss2 / self.encoder.num_classifiers
        loss3 = loss3 / self.encoder.num_classifiers
        #print (loss_data)
        #stats = self.stats(loss_data, scores.data, target.data)
        stats = self.stats_all(loss_data, scores.data, target.data, loss_data2, labels.data, target2.data)

        return loss, stats, loss2, loss3


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute.make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
