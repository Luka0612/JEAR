# coding: utf-8
import numpy as np
import os
import sys
import yaml
import pickle

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))
sys.path.append(current_relative_path(".."))
from utils.utils import getRelID, getEntityID, getMatrixForContext, readIndices


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        conf = yaml.load(f)
    return conf


def process_data_file(data_file):
    data_in = open(data_file, 'rb')
    train_id2sent = pickle.load(data_in)
    train_id2pos = pickle.load(data_in)
    train_id2ner = pickle.load(data_in)
    train_id2nerBILOU = pickle.load(data_in)
    train_id2arg2rel = pickle.load(data_in)

    test_id2sent = pickle.load(data_in)
    test_id2pos = pickle.load(data_in)
    test_id2ner = pickle.load(data_in)
    test_id2nerBILOU = pickle.load(data_in)
    test_id2arg2rel = pickle.load(data_in)
    data_in.close()
    return (train_id2sent, train_id2pos, train_id2ner, train_id2nerBILOU, train_id2arg2rel), \
           (test_id2sent, test_id2pos, test_id2ner, test_id2nerBILOU, test_id2arg2rel)


def split_context(curId, id2ner, id2arg2rel):
    cur_ners = id2ner[curId].split()
    cur_rel = id2arg2rel[curId]

    entities = []
    i = 0
    while i < len(cur_ners):
        j = i + 1
        while j < len(cur_ners) and cur_ners[i].split("-")[-1] == cur_ners[j].split("-")[-1]:
            j += 1
        if cur_ners[i] != "O":
            entities.append((i, j - 1))
        i = j

    d_entities_rels_index = {}
    for e1Ind in range(len(entities)-1, -1, -1):
        entities_rels = []
        ent1 = entities[e1Ind]
        for e2Ind in range(e1Ind - 1, -1, -1):
            ent2 = entities[e2Ind]
            if (ent2[1], ent1[1]) in cur_rel:
                entities_rels.append(getRelID(cur_rel[(ent2[1], ent1[1])]))
            else:
                entities_rels.append(0)
        entities_rels.reverse()
        d_entities_rels_index[ent1] = entities_rels

    cur_ners_index = [getEntityID(i) for i in cur_ners]
    return cur_ners_index, d_entities_rels_index


def process_samples(id2sent, id2ner, id2arg2rel, wordindices, conf):
    all_data = []
    for curId in id2sent:
        context = id2sent[curId]
        context_index = getMatrixForContext(context.split(), conf['contextsize'], wordindices)
        cur_ners_index, d_entities_rels_index = split_context(curId, id2ner, id2arg2rel)

        if len(cur_ners_index) >= conf['contextsize']:
            cur_ners_index = np.array(cur_ners_index)[:conf['contextsize']]
        else:
            matrix = np.zeros(shape=(conf['contextsize'],))
            matrix[:len(cur_ners_index)] = np.array(cur_ners_index)
            cur_ners_index = matrix

        one_data = np.array([context_index, cur_ners_index])
        all_data.append(one_data)
    return all_data


def data_process(conf):
    data_file = conf["datafile"]
    train_data, test_data = process_data_file(data_file)
    word_vectors_file = conf["wordvectors"]
    wordindices = readIndices(word_vectors_file, isWord2vec = True)
    process_samples(train_data[0], train_data[3], train_data[4], wordindices, conf)


def main():
    conf = load_yaml(current_relative_path("../../conf/config.yaml"))
    data_process(conf)

if __name__ == '__main__':
    main()