#!/usr/bin/env python3
import os
import pandas as pd
from keras.preprocessing import text, sequence

current_dir = os.path.dirname(os.path.realpath(__file__))

# 對聯的資料處理
def app_1_handler():
    data_dir = os.path.join(current_dir, "..", "data/app_1")
    tr_in = pd.read_csv(os.path.join(data_dir, "train", "in.txt"), sep="\t", header=None)[0].tolist()
    tr_out = pd.read_csv(os.path.join(data_dir, "train", "out.txt"), sep="\t", header=None)[0].tolist()
    te_in = pd.read_csv(os.path.join(data_dir, "test", "in.txt"), sep="\t", header=None)[0].tolist()
    te_out = pd.read_csv(os.path.join(data_dir, "test", "out.txt"), sep="\t", header=None)[0].tolist()

    # 生成的字典取频数高的max_feature个word对文本进行处理，其他的word都会忽略
    max_feature = 10000
    # 每个文本处理后word长度为maxlen
    maxlen = 50

    # 使用keras进行分词，word转成对应index
    tokenizer = text.Tokenizer(num_words=max_feature)
    tokenizer.fit_on_texts(tr_in + tr_out + te_in + te_out)
    tr_f = tokenizer.texts_to_sequences(tr_in)
    te_f = tokenizer.texts_to_sequences(te_in)
    # 将每个文本转成固定长度maxlen，长的截取，短的填充0
    tr_f = sequence.pad_sequences(train_feature, maxlen)
    te_f = sequence.pad_sequences(test_feature, maxlen)



if __name__ == '__main__':
    app_1_handler()
