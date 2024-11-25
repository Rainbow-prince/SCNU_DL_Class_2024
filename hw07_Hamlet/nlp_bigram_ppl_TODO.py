import os

import sentencepiece as spm
import numpy as np
from collections import defaultdict
from typing import List
import math

import nltk  # Natural Language Toolkit
from nltk.util import ngrams  # utility 实用工具
from collections import Counter

nltk.download('punkt')

# debug = False
debug = True
###############################################
# generate Bi-Gram counter for training corpus
###############################################
corpus_text = ''' I play tennis. I like Chinese friends. I talk with Chinese student. I play with tennis friends. I have friends who like tennis.'''
token = nltk.word_tokenize(corpus_text)

###########################
# Generate Bi-Gram counter
###########################
# 统计单个token（unigram）的出现次数
unigrams = Counter([w[0] for w in list(ngrams(token, 1))])
# 统计各个 bigrams 的出现次数
bigrams = Counter(list(ngrams(token, 2)))


###########################
# generate query Bi-Gram
###########################
query_text_1 = "I play with Chinese friends"
query_text_2 = "Chinese friends who like tennis"
query_token = nltk.word_tokenize(query_text_1)
query_bigram = list(ngrams(query_token, 2))

# DO NOT MODIFY ABOVE


if debug:
    print(token, end='\n\n')
    print(unigrams, end='\n\n')
    print(bigrams, end='\n\n')
    print(query_bigram, end='\n\n')

###########################
# TODO: lookup each query bigram in each query_text
# compute Uni-Counter[bg[0]] /  Bi-Counter[(bg[0],bg[1])]
# convert to PPL and output

def calculate_ppl1(bigrams: Counter, unigrams: Counter, query_bigrams: list)->float:
    """
    计算ppl，基于对数概率的实现
    @param bigrams: 包含训练语料库中所有 Bi-Gram 的出现次数
    @param unigrams:  包含训练语料库中所有 Unigram 的出现次数
    @param query_bigrams: 查询语句的 bi-gram
    @return:
    """
    total_prob = 0
    for bigram in query_bigrams:
        # 获取 Bigram & Unigram 计数使用拉普拉斯平滑避免除以零
        count_bi = bigrams.get(bigram, 0) + 1
        count_uni = unigrams.get(bigram[0], 0) + len(unigrams)
        prob = count_bi / count_uni  # 计算概率
        if prob == 0:  # 避免对数计算错误
            prob = 1e-10  # 设置一个非常小的正数
        total_prob += math.log(prob)
    ppl = math.exp(-total_prob / len(query_bigrams))
    return ppl



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


# Tokenize the second query
query_token_2 = nltk.word_tokenize(query_text_2)
query_bigram_2 = list(ngrams(query_token_2, 2))

# 计算ppl
ppl_query_1 = calculate_ppl(bigrams, unigrams, query_bigram)
ppl_query_2 = calculate_ppl(bigrams, unigrams, query_bigram_2)
print(f"PPL for '{query_text_1}': {ppl_query_1:.2f}")
print(f"PPL for '{query_text_2}': {ppl_query_2:.2f}")
