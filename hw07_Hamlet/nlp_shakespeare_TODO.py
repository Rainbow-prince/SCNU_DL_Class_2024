import os
import nltk
import numpy as np
from typing import List
from transformers import AutoTokenizer
from nltk.util import ngrams
from collections import Counter
import math

# TODO
# download tokenizer from https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main
# tokenizer.json
# special_tokens_map.json
# tokenizer_config.json
# put them in ./model
tokenizer = AutoTokenizer.from_pretrained('./model')

#####################################
########## Do not modify ###########
#####################################
corpus = []
for file_name in os.listdir('shakespeare-db'):
    #print(file_name)
    with open(os.path.join('shakespeare-db',file_name)) as file:
        corpus += [line.strip() for line in file]
# tokenize sentences into tokens
# do not modify
sent_all = []
for text in corpus:
    words = tokenizer.tokenize(text, add_special_tokens=True)
    sent_all += words


###########################
# Generate Bi-Gram counter
###########################
unigrams = Counter([w[0] for w in list(ngrams(sent_all, 1))])
bigrams = Counter(list(ngrams(sent_all, 2)))
# tokenize THE QUESTION
the_question = "To be, or not to be: that is the question"
the_question_tokens = tokenizer.tokenize(the_question)

# TODO: compute PPL here

def calculate_ppl(bigrams: Counter, unigrams: Counter, query_bigrams: List[tuple]) -> float:
    """
    计算 PPL（Perplexity）——基于概率的定义
    @param bigrams: 包含训练语料库中所有 Bi-Gram 的出现次数
    @param unigrams: 包含训练语料库中所有 Unigram 的出现次数
    @param query_bigrams: 查询语句的 Bi-Gram 列表
    @return: PPL 值
    """
    total_prob = 1  # 初始化总概率为 1，要算乘积

    # 计算一系列概率，后面再根据此计算ppl
    for bigram in query_bigrams:
        # 获取 Bigram & Unigram 计数使用拉普拉斯平滑避免除以零（拉普拉斯似乎有问题）
        count_bi = bigrams.get(bigram, 0) + 0.1  # 如果没出现过，返回0
        count_uni = unigrams.get(bigram[0], 0) + len(unigrams)

        prob = count_bi / count_uni  # 计算概率
        total_prob *= prob  # 乘以当前 Bi-Gram 的概率

    # 计算 PPL
    ppl = 1 / math.pow(total_prob, 1.0 / len(query_bigrams))
    return ppl

# Calculate PPL for the question
ppl = calculate_ppl(bigrams, unigrams, the_question_tokens)
print(f"ppl for “{the_question}”: {ppl:.2f}")