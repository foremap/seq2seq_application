#!/usr/bin/env python3
from model import Model

def couplet():
    m = Model(
        '../data/couplet/train/in.txt',
        '../data/couplet/train/out.txt',
        '../data/couplet/test/in.txt',
        '../data/couplet/test/out.txt',
        '../data/couplet/vocabs',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='./output',
        restore_model=False)

    m.train(100000)

if __name__ == '__main__':
    couplet()