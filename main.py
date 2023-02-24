import gc
import os

import argparse
from numpy.random import seed
import sqlite3
import numpy as np
import pandas as pd
import tensorflow.keras.backend as k
import util
import data_processing
from myconfig import Model_config
import random

config = Model_config()

def main(args):
    seed = 0
    CV = 5
    interaction_num = 10
    conn = sqlite3.connect("data/event.db")
    drug_info = pd.read_sql('select * from drug;', conn)
    feature_list = args['featureList']
    featureName = "+".join(feature_list)

    model_list = args['model']
    model_name = "+".join(model_list)
    model_name = model_name[:-1]

    result_all = dict()
    result_eve = dict()
    all_matrix = list()

    drug_list = []
    for line in open("data/DrugList.txt", 'r'):
        drug_list.append(line.split()[0])

    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    save_data = dict()

    label_path = "label_data/"
    data_filename = label_path + config.label_data_filename
    data_label = None
    event_num = 0
    feature_dim = []

    if not os.path.exists(data_filename):
        for feature in feature_list:
            print(feature)
            data_feature, data_label, event_num = data_processing.prepare(drug_info,
                                                                          feature,
                                                                          config.vector_size,
                                                                          mechanism,
                                                                          action,
                                                                          drugA,
                                                                          drugB)
            save_data[feature] = data_feature
            feature_dim.append(data_feature.shape[1]//2)
            all_matrix.append(data_feature)

        save_data["label"] = data_label
        save_data["event_num"] = event_num

        np.save(data_filename, save_data)
    else:
        data_dic = np.load(data_filename, allow_pickle=True).item()
        for feature in feature_list:
            print(feature)
            all_matrix.append(data_dic[feature])
            data_feature = data_dic[feature]
            feature_dim.append(data_feature.shape[1] // 2)

        data_label = data_dic["label"]
        event_num = int(data_dic["event_num"])

    for model in model_list:
        print(model)
        import time
        start = time.process_time()
        all_result, each_result = util.cross_validation(all_matrix,
                                                        data_label,
                                                        model,
                                                        event_num,
                                                        seed,
                                                        CV,
                                                        model_name,
                                                        feature_dim)
        print(all_result)
        end = time.process_time()
        print('Running time of {}: {} Seconds'.format(model, end - start))
        util.save_result(featureName, 'all', model, all_result)
        result_all[model] = all_result
        result_eve[model] = each_result
        del all_result
        del each_result

    gc.collect()
    k.clear_session()

if __name__ == "__main__":
    DDIADN_KGE_feature = ["smile",
                          "target",
                          "enzyme",
                          "kg_embedding",
                          "pathway"]
    best_combine = ["target", "enzyme", "kg_embedding"]
    KGDDI_feature = ["kg_embedding"]
    DeepDDI_feature = ["smile"]
    from itertools import combinations
    for i in range(len(DDIADN_KGE_feature), len(DDIADN_KGE_feature)+1):
        i_combine_list = list(combinations(DDIADN_KGE_feature, i))
        for cur_feature in i_combine_list:
            print(list(cur_feature))
            parser = argparse.ArgumentParser()
            parser.add_argument("-f",
                                "--featureList",
                                default=cur_feature,
                                help="features to use",
                                choices=["target",
                                         "enzyme",
                                         "smile",
                                         "kg_embedding",
                                         "pathway"],
                                nargs="+")
            parser.add_argument("-c", "--model",
                                choices=["MMADL_DDI",
                                         "DDKG_DDI",
                                         "KG_DDI"
                                         "MDL_DDI"
                                         "Deep_DDI"],
                                default=["MMADL_DDI"],
                                help="model to use",
                                nargs="+")

            args = vars(parser.parse_args())

            for i in range(0, 5):
                print(args)
                main(args)
                gc.collect()
                break
            break
        # break


