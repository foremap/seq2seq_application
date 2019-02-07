from collections import defaultdict
from collections import Counter
from string import punctuation
import numpy as np
import os
import pdb


def filter_text(text):
    return ''.join([c for c in text if c not in punctuation]).lower()

def preprocess_text(file_path_src, file_path_tar, max_feats):
    f_src = open(file_path_src)
    f_tar = open(file_path_tar)
    vocab = defaultdict(int)
    freq_src = defaultdict(int)
    freq_tar = defaultdict(int)
    sents_src = [line.rstrip() for line in f_src.readlines()]
    sents_tar = [line.rstrip() for line in f_tar.readlines()]

    for sent in sents_src:
        sent = filter_text(sent)
        for word in sent.split(" "):
            freq_src[word] += 1

    for sent in sents_tar:
        for word in sent.split(" "):
            freq_tar[word] += 1

    freq_sorted_src = Counter(freq_src)
    freq_sorted_src = freq_sorted_src.most_common(max_feats)
    freq_sorted_tar = Counter(freq_tar)
    freq_sorted_tar = freq_sorted_tar.most_common(max_feats)

    freq_words_src = list(map(lambda x:x[0], freq_sorted_src))
    freq_words_tar = list(map(lambda x:x[0], freq_sorted_tar))

    vocab_src = dict({v:k+1 for k, v in enumerate(freq_words_src)})
    vocab_tar = dict({v:k+1 for k, v in enumerate(freq_words_tar)})

    vocab_src["UNK"] = max_feats
    vocab_src["<s>"] = max_feats + 1
    vocab_src["</s>"] = max_feats + 2
    vocab_src["PAD"] = max_feats + 3

    vocab_tar["UNK"] = max_feats
    vocab_tar["<s>"] = max_feats + 1
    vocab_tar["</s>"] = max_feats + 2
    vocab_tar["PAD"] = max_feats + 3
    #pdb.set_trace()
    return vocab_src, vocab_tar, sents_src, sents_tar


def text2seq_generator(vocab_src, vocab_tar, sents_src, sents_tar):
    unk_key = vocab_src["UNK"]
    for sent_src, sent_tar in zip(sents_src, sents_tar):
        seq_src = list(map(lambda x:vocab_src.get(x, unk_key), filter_text(sent_src).split(" ")))
        seq_tar = list(map(lambda x:vocab_tar.get(x, unk_key), sent_tar.split(" ")))
        #pdb.set_trace()
        yield seq_src, seq_tar


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, "..", "data/couplet", "train")

    vs, vt, ss, st = preprocess_text(
                        os.path.join(data_dir, "in.txt"), 
                        os.path.join(data_dir, "out.txt"), 
                        5000
                    )
    for x, y in text2seq_generator(vs, vt, ss, st):
       print(y)
       print(seq2text_generator(y, vt))
