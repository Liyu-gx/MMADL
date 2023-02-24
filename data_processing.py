# 基于特征 计算相识度的函数 of two durg
import os

import numpy as np

import util
from sklearn.decomposition import PCA
import pandas as pd
import collections
from myconfig import Model_config

config = Model_config()


def prepare(drug_inf, feature_name, vector_size, mechanism, action, drugA, drugB):

    d_label = dict()
    d_feature = dict()

    # Splice the features
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    d_event_count = collections.Counter(d_event)
    event_count_list = sorted(d_event_count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(event_count_list)):
        d_label[event_count_list[i][0]] = i

    drug_list = np.array(drug_inf['name']).tolist()
    feature_save_name = feature_name + ("_row" if config.is_row_feature else "_sim")
    feature_save_path = "drug_to_feature/" + "drug_" + feature_save_name + ".npy"
    if os.path.exists(feature_save_path):
        d_feature = np.load(feature_save_path, allow_pickle=True).item()
    else:
        vector = feature_vector(feature_name, drug_inf, vector_size, is_row=config.is_row_feature)
        # Transfrom the drug ID to feature vector
        for i in range(len(drug_list)):
            # drug_id : feature_vector
            d_feature[drug_list[i]] = vector[i]

        np.save(feature_save_path, d_feature)

    new_feature = list()
    new_label = list()
    name_to_id = dict()
    for i in range(len(d_event)):
        new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]])))
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    print(new_feature.shape)

    return new_feature, new_label, config.event_num

def feature_vector(feature_name, drug_inf, vector_size, is_row):
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(
            np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    if feature_name == "kg_embedding":
        embedding = util.get_or_embedding(config.embedding_path, config.embedding_size, drug_inf)
        pca = PCA(n_components=config.embedding_size)
        pca.fit(embedding)
        embedding_matrix = pca.transform(embedding)
        return embedding_matrix

    feature_set = set()
    drug_feature_list = np.array(drug_inf[feature_name]).tolist()
    for i in drug_feature_list:
        for feature in i.split('|'):
            if feature not in feature_set:
                feature_set.add(feature)

    # 初始化特征矩阵
    feature_matrix = np.zeros(shape=(len(drug_feature_list), len(feature_set)),
                              dtype=float)
    df_feature_matrix = pd.DataFrame(feature_matrix,
                              columns=list(feature_set))

    for drug_i in range(len(drug_feature_list)):
        for feature in drug_feature_list[drug_i].split("|"):
            df_feature_matrix[feature].iloc[drug_i] = 1
    if is_row:
        return np.array(df_feature_matrix)
    else:
        sim_matrix = Jaccard(np.array(df_feature_matrix))
        pca = PCA(n_components=vector_size)
        pca.fit(sim_matrix)
        sim_matrix = pca.transform(sim_matrix)
        return sim_matrix
