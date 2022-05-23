from __future__ import absolute_import
from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger','FileLogger', 'LoggerMonitor', 'savefig', 'TensorboardLogger']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)

def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]



#based on https://groups.google.com/forum/#!topic/comp.lang.python/0lqfVgjkc68
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        bufsize = 1
        self.log = open(filename, "w", buffering=bufsize)

    def delink(self):
        self.log.close()
        self.log = open('foo', "w")

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def isatty(self):
        return True


class FileLogger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


class TensorboardLogger:
  def __init__(self, output_dir, is_master=False):
    self.output_dir = output_dir
    self.current_step = 0
    if is_master: self.writer = SummaryWriter(self.output_dir)
    else: self.writer = NoOp()
    self.log('first', time.time())

  def log(self, tag, val):
    """Log value to tensorboard (relies on global_example_count being set properly)"""
    if not self.writer: return
    self.writer.add_scalar(tag, val, self.current_step)

  def update_step_count(self, batch_total):
    self.current_step += batch_total

  def close(self):
    self.writer.export_scalars_to_json(self.output_dir+'/scalars.json')
    self.writer.close()

  def flush(self):
    if not self.writer: return
    self.writer.flush()

  # Convenience logging methods
  def log_size(self, bs=None, sz=None):
    if bs: self.log('sizes/batch', bs)
    if sz: self.log('sizes/image', sz)

  def log_acc_loss(self, dic):
    for k,v in dic.items():
        self.log("losses/%s"%k, v)

  def log_grad_hist(self, dic):
    for k,v in dic.items():
        self.writer.add_histogram("mask_grads/%s"%k, v.cpu().detach().type(torch.float32), self.current_step)

  def log_memory(self):
    if not self.writer: return
    self.log("memory/allocated_gb", torch.cuda.memory_allocated()/1e9)
    self.log("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9)
    self.log("memory/cached_gb", torch.cuda.memory_cached()/1e9)
    self.log("memory/max_cached_gb", torch.cuda.max_memory_cached()/1e9)

  def log_trn_times(self, batch_time, data_time, batch_size):
    if not self.writer: return
    self.log("times/step", 1000*batch_time)
    self.log("times/data", 1000*data_time)
    images_per_sec = batch_size/batch_time
    self.log("times/1gpu_images_per_sec", images_per_sec)
    self.log("times/8gpu_images_per_sec", 8*images_per_sec)

if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
    'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt',
    'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    }

    field = ['Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')
