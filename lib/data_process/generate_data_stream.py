# coding: utf-8
import os
import pickle
import sys

import numpy as np
import yaml

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))
sys.path.append(current_relative_path(".."))
from utils.utils import getRelID, getEntityID, getMatrixForContext, readIndices

rel_entity_num = 31
entity_rel_num = 30


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


def id_to_one_hot(id_index):
    one_hot = np.zeros((18,), dtype=np.int32)
    one_hot[id_index] = 1
    return one_hot


def split_context(curId, id2ner, id2arg2rel):
    cur_ners = id2ner[curId].split()
    cur_rel = id2arg2rel[curId]
    valid_entity_rel_num = []

    entities = []
    i = 0
    while i < len(cur_ners):
        j = i + 1
        while j < len(cur_ners) and cur_ners[i].split("-")[-1] == cur_ners[j].split("-")[-1]:
            j += 1
        if cur_ners[i] != "O":
            entities.append((i, j - 1))
        i = j

    rels_entities_index = []
    np_entities_rels_all = []
    np_entities_ner_rels_all = []
    for e1Ind in range(len(entities) - 1, -1, -1):
        if e1Ind == 0:
            continue
        entities_rels = []
        entities_ner_rels = []
        ent1 = entities[e1Ind]
        for e2Ind in range(e1Ind - 1, -1, -1):
            ent2 = entities[e2Ind]
            for ent2_index in range(ent2[1], ent2[0] - 1, -1):
                if (ent2[1], ent1[1]) in cur_rel:
                    entities_rels.append(getRelID(cur_rel[(ent2[1], ent1[1])]))
                else:
                    entities_rels.append(1)
                entities_ner_rels.append(ent2_index)
        entities_rels.reverse()
        entities_ner_rels.reverse()

        np_entities_rels = np.zeros(shape=(entity_rel_num,), dtype=np.int32)
        np_entities_rels[:len(entities_rels)] = np.array(entities_rels, dtype=np.int32)

        np_entities_ner_rels = np.ones(shape=(entity_rel_num,), dtype=np.int32) * 120
        np_entities_ner_rels[:len(entities_ner_rels)] = np.array(entities_ner_rels, dtype=np.int32)

        for ent1_index in range(ent1[1], ent1[0] - 1, -1):
            valid_entity_rel_num.append(len(entities_rels))
            rels_entities_index.append(ent1_index)
            np_entities_rels_all.append(np_entities_rels)
            np_entities_ner_rels_all.append(np_entities_ner_rels)

    cur_ners_index_one_hot = np.array([id_to_one_hot(getEntityID(i)) for i in cur_ners])
    cur_ners_index = np.array([getEntityID(i) for i in cur_ners])

    for additional in range(len(rels_entities_index), rel_entity_num):
        valid_entity_rel_num.append(0)
        rels_entities_index.append(120)
        np_entities_ner_rels = np.ones(shape=(entity_rel_num,), dtype=np.int32) * 120
        np_entities_ner_rels_all.append(np_entities_ner_rels)

        np_entities_rels = np.zeros(shape=(entity_rel_num,), dtype=np.int32)
        np_entities_rels_all.append(np_entities_rels)

    return cur_ners_index_one_hot, cur_ners_index, (
        np.array(rels_entities_index, dtype=np.int32), np.array(valid_entity_rel_num, dtype=np.int32),
        np.array(np_entities_rels_all, dtype=np.int32),
        np.array(np_entities_ner_rels_all, dtype=np.int32))


def process_samples(id2sent, id2ner, id2arg2rel, wordindices, conf):
    all_data = []
    for curId in id2sent:
        context = id2sent[curId]
        context_index = getMatrixForContext(context.split(), conf['contextsize'], wordindices)
        cur_ners_index_one_hot, cur_ners_index, entities_rels_index_tuple = split_context(curId, id2ner, id2arg2rel)

        if len(cur_ners_index) >= conf['contextsize']:
            cur_ners_index_one_hot = np.array(cur_ners_index_one_hot)[:conf['contextsize']]
            cur_ners_index = np.array(cur_ners_index)[:conf['contextsize']]
        else:
            matrix = np.zeros(shape=(conf['contextsize'],))
            matrix[:len(cur_ners_index)] = np.array(cur_ners_index)
            cur_ners_index = matrix

            matrix = np.zeros(shape=(conf['contextsize'], 18,))
            matrix[:len(cur_ners_index_one_hot)] = np.array(cur_ners_index_one_hot)
            cur_ners_index_one_hot = matrix

        one_data = (context_index, cur_ners_index_one_hot, cur_ners_index, entities_rels_index_tuple)
        all_data.append(one_data)

    return all_data


def data_process(conf):
    data_file = conf["datafile"]
    train_data, test_data = process_data_file(data_file)
    word_vectors_file = conf["wordvectors"]
    wordindices = readIndices(word_vectors_file, isWord2vec=True)
    return process_samples(train_data[0], train_data[3], train_data[4], wordindices, conf)


def generate_batch(Tconf):
    conf = {}
    conf["datafile"] = Tconf.datafile
    conf["wordvectors"] = Tconf.wordvectors
    conf["contextsize"] = Tconf.contextsize
    all_data = data_process(conf)
    all_data = all_data[
               int(len(all_data) * Tconf.start_split_data_index): int(len(all_data) * Tconf.end_split_data_index)]
    for i in range(int(len(all_data) / Tconf.batch_size)):
        batch = all_data[i: i + Tconf.batch_size]
        batch_inputs = np.array([i[0] for i in batch])
        batch_labels = np.array([i[1] for i in batch])
        batch_labels_index = np.array([i[2] for i in batch])
        batch_rel_entity_index = np.array([i[3][0] for i in batch])
        batch_rel_entity_index_match = np.array([i[3][3] for i in batch])
        train_rel = np.array([i[3][2] for i in batch])
        rel_masks = np.array([i[3][1] for i in batch])
        yield batch_inputs, batch_labels, batch_labels_index, batch_rel_entity_index, batch_rel_entity_index_match, train_rel, rel_masks
    yield None, None, None, None, None, None, None


def main():
    conf = load_yaml(current_relative_path("../../conf/config.yaml"))
    data_process(conf)


if __name__ == '__main__':
    class Conf:
        # 词向量维度
        num_steps = 128
        # batch的大小
        batch_size = 1
        entity_class_num = 18
        vocabulary_size = 418129
        embedding_size = 128
        keep_prob = 0.9
        encode_hidden_size = 128
        encode_num_layers = 4
        decode_embedding_dim = 128
        decode_hidden_size = 128
        decode_num_layers = 4
        max_max_epoch = 1000
        # 学习率
        lr = 0.8
        # 学习率衰减
        lr_decay = 0.8
        max_epoch = 5
        datafile = current_relative_path("../../data/corpus_prepared.pickled")
        wordvectors = current_relative_path("../../data/vecs.lc.over100freq.txt.gz")
        contextsize = 120
        start_split_data_index = 0.0
        end_split_data_index = 0.8


    a = generate_batch(Conf)
    print a.next()[4].shape
