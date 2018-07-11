# coding: utf-8
import gzip
import re

import numpy as np


def getRelID(relName):
    relSet = ['<pad>', 'O', 'OrgBased_In', 'Live_In', 'Kill', 'Located_In', 'Work_For']
    return relSet.index(relName)


def getEntityID(entityName):
    nerSet = ['<pad>', 'O', 'L-Org', 'U-Loc', 'U-Peop', 'U-Org', 'B-Org', 'B-Other', 'I-Org', 'B-Peop', 'I-Loc', 'I-Peop',
              'I-Other', 'L-Loc', 'U-Other', 'L-Other', 'B-Loc', 'L-Peop']
    return nerSet.index(entityName)


def getMatrixForContext(context, contextsize, wordindices):
    matrix = np.zeros(shape=(contextsize), dtype=np.int32)
    i = 0
    while i < len(context):
        word = context[i]
        if word != "<empty>":
            if not word in wordindices:
                if re.search(r'^\d+$', word):
                    word = "0"
                if word.islower():
                    word = word.title()
                else:
                    word = word.lower()
            if not word in wordindices:
                word = "<unk>"
            curIndex = wordindices[word]
            matrix[i] = curIndex
        i += 1

    return matrix


def readIndices(wordvectorfile, isWord2vec=True):
    indices = {}
    curIndex = 0
    indices["<empty>"] = curIndex
    curIndex += 1
    indices["<unk>"] = curIndex
    curIndex += 1
    if ".gz" in wordvectorfile:
        f = gzip.open(wordvectorfile, 'r')
    else:
        f = open(wordvectorfile, 'r')
    count = 0
    for line in f:
        if isWord2vec:
            if count == 0:
                print "omitting first embedding line because of word2vec"
                count += 1
                continue
        parts = line.split()
        word = parts[0]
        indices[word] = curIndex
        curIndex += 1
    f.close()
    return indices
