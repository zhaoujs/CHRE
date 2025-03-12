#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''

'''
import copy
import math
import os
from collections import Counter

import numpy as np
from numpy import NAN
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from itertools import chain

class Data_Entropy:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pred = NAN
        self.pred_prob = NAN
        self.source = NAN




class cg_Global():
    def __init__(self) -> None:
        self.model = NAN
        self.old_model = NAN
        self.Entropy_list = NAN
        self.K = NAN
        self.RKNN_MAX_COUNT = 0
        self.lamba_aim1 = 0.8
        self.lamba_aim2 = 0.5
        self.Pseudo_num = 0
        self.model_list = []
        self.No_T = 0
        self.Data_list = []
        self.noise_list = []
        self.filename = None
        self.noise_level = None

def KNN_nei_Matrix_Pseudo(data, test, K, is_include=True):
    """
    :param data:
    :param test:
    :param K:
    :return:
    """
    if is_include:
        knn = KNN(k=K + 1)
        knn.fit(data)
        RKNN_Matrix = []
        RKNN_SAMPLES = []
        for item in test:
            _, samples, ls_res = knn.predict(item)
            RKNN_Matrix.append(ls_res[1:])
            RKNN_SAMPLES.append(samples[1:])
        return RKNN_Matrix, RKNN_SAMPLES
    else:
        knn = KNN(k=K)
        knn.fit(data)
        RKNN_Matrix = []
        RKNN_SAMPLES = []
        for item in test:
            _, samples, ls_res = knn.predict(item)
            RKNN_Matrix.append(ls_res)
            RKNN_SAMPLES.append(samples)
        return RKNN_Matrix, RKNN_SAMPLES


def CG_CalPseudo(Pseudo_Entropy, test_D, Global_item):
    lamba_R = Global_item.lamba_aim1
    lamba_U = Global_item.lamba_aim2
    No = Global_item.No_T
    if No == 0:
        Pseudo_Entropy = dict_normal(Pseudo_Entropy, "SInf")
        Pseudo_Entropy = dict_normal(Pseudo_Entropy, "SNoi")
        for item in Pseudo_Entropy:
            item["value"] = (1 - lamba_U) * item["SInf"] - lamba_U * item["SNoi"]
    else:
        Model_before = Global_item.model_list[-2]
        Model_curr = Global_item.model_list[-1]
        before_prob = Model_before.predict_proba(test_D.x)
        curr_prob = Model_curr.predict_proba(test_D.x)

        for item in Pseudo_Entropy:
            item["DInf"] = abs(before_prob[item["index"]][0] - curr_prob[item["index"]][0])
            item["DNoi"] = curr_prob[item["index"]][0]
        Pseudo_Entropy = dict_normal(Pseudo_Entropy, "SInf")
        Pseudo_Entropy = dict_normal(Pseudo_Entropy, "SNoi")
        Pseudo_Entropy = dict_normal(Pseudo_Entropy, "DNoi")
        Pseudo_Entropy = dict_normal(Pseudo_Entropy, "DInf")
        for item in Pseudo_Entropy:
            item["Content_Inf"] = (1 - lamba_R) * item["SInf"] + lamba_R * item["DInf"]
            item["Content_Noi"] = (1 - lamba_R) * item["SNoi"] + lamba_R * item["DNoi"]
            item["value"] = (1 - lamba_U) * item["Content_Inf"] - lamba_U * item["Content_Noi"]

def Entropy(k=5, Same_class=3):
    same = Same_class
    diff = k - same
    res = -1
    pianzhi = 0.001
    prob_same = (same + pianzhi) / (k + pianzhi)
    prob_difff = (diff + pianzhi) / (k + pianzhi)
    if same / k < 0.5:
        res = - prob_same * math.log(prob_same) - prob_difff * math.log(prob_difff)
        res = 2 * (-math.log(0.5)) - res
    else:
        res = - prob_same * math.log(prob_same) - prob_difff * math.log(prob_difff)
    return res

def Entropy_all_MinMax(k=5, min_num: float = 0, max_num: float = 1):
    res = []
    for i in range(k + 1):
        res.append(Entropy(k=k, Same_class=i))
    arr = np.array(res)
    min_val = np.min(arr)
    max_val = np.max(arr)
    return min_num + (max_num - min_num) * (arr - min_val) / (max_val - min_val)

def loadfile(filename):
    x, y = load_svmlight_file(filename)
    return np.array(x.todense()), np.array(y.astype(np.int))


def Stratified_fold_K_version_2(x, y, n_spli=5, random_sta=42):
    skf = StratifiedKFold(n_splits=n_spli, shuffle=True, random_state=random_sta)
    i = 0
    res = {}
    train_x = []
    train_y = []
    test_x = []
    test_y =[]
    for train_index, test_index in skf.split(x, y):
        i = i + 1
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_x.append(copy.deepcopy(X_train))
        train_y.append(copy.deepcopy(y_train))
        test_x.append(copy.deepcopy(X_test))
        test_y.append(copy.deepcopy(y_test))
    res["train_x"] = train_x
    res["train_y"] = train_y
    res["test_x"] = test_x
    res["test_y"] = test_y
    return res

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=self.k)

    def fit(self, X):
        self.X = X
        self.model.fit(X)

    def predict(self, sample):
        distances, indices = self.model.kneighbors(np.array(sample).reshape(1, -1))
        indices = indices[0]
        samples = [self.X[i] for i in indices]
        return distances[0], samples, indices



def RKNN_nei(RKNN_Matrix, idx):
    RKNN_res = []
    for i, item in enumerate(RKNN_Matrix):
        if idx in item:
            RKNN_res.append(i)
    return RKNN_res


def KnnAndRknn(train_D, K=5, Gloabl_item=None):
    data = train_D.x
    y = train_D.y
    Entropy_list = Gloabl_item.Entropy_list
    res = []
    KNN_Matrix, KNN_SAMPLES = KNN_nei_Matrix(data, K)
    for index, item in enumerate(data):
        RKNN_idx = RKNN_nei(KNN_Matrix, index)
        KNN_idx = KNN_Matrix[index]
        ls_dict = {}
        ls_dict["index"] = index
        ls_dict["y_true"] = y[index]
        ls_dict["KNN"] = KNN_idx
        ls_dict["KNN_sample"] = y[KNN_idx]
        ls_dict["RKNN"] = RKNN_idx
        ls_dict["RKNN_sample"] = y[RKNN_idx]
        ls_dict["KNN_detail"] = KNN_SAMPLES[index]
        ls_dict["RKNN_detail"] = data[RKNN_idx]
        ls_dict["curr_item"] = item

        ls_dict["SInf"] = 0
        curr_count = Counter(ls_dict["KNN_sample"])
        curr_label = ls_dict["y_true"]
        curr_Entropy = Entropy_list[curr_count[curr_label]]
        ls_dict["SInf"] = curr_Entropy
        ls_dict["SNoi"] = 0
        RKNN_sam = ls_dict["RKNN_sample"]
        Rknn_len = len(RKNN_sam)

        if Rknn_len == 0:
            ls_dict["noise"] = 0
        else:
            curr_count = Counter(RKNN_sam)
            this_list = Entropy_all_MinMax(k=Rknn_len, min_num=0.0, max_num=1)
            RKNN_samm = curr_count[curr_label]
            gap = -1
            if RKNN_samm != Rknn_len:
                gap = this_list[RKNN_samm] - this_list[RKNN_samm + 1]
            else:
                gap = this_list[RKNN_samm] - 0
            nois_Entropy = this_list[RKNN_samm] - gap * RKNN_samm / (Rknn_len + 1)
            ls_dict["SNoi"] = nois_Entropy
            ls_dict["noise"] = nois_Entropy ** (1 / (Rknn_len - curr_count[curr_label] + 1))
            ls_dict["noise2"] = nois_Entropy ** (curr_count[curr_label])



        if Gloabl_item.noise_list != None:
            if index in Gloabl_item.noise_list:
                ls_dict["add_noise"] = 1
            else:
                ls_dict["add_noise"] = 0
        res.append(ls_dict)
    return res


def dict_normal(dict_item, value_item, Min=0, Max=1):
    values = [item["{}".format(value_item)] for item in dict_item]
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        for item in dict_item:
            item["{}".format(value_item)] = min_value
    else:
        for item in dict_item:
            ls = item["{}".format(value_item)]
            item["{}".format(value_item)] = (Max - Min) * (ls - min_value) / (max_value - min_value)
    return dict_item

def cal_target_value(nn_list, train_D, Global_entropy):
    lamba_R = Global_entropy.lamba_aim1
    lamba_U = Global_entropy.lamba_aim2
    No = Global_entropy.No_T
    if No <= 0:
        nn_list = dict_normal(nn_list, "SInf")
        nn_list = dict_normal(nn_list, "SNoi")
        for item in nn_list:
            item["value"] = (1 - lamba_U) * item["SInf"] - lamba_U * item["SNoi"]
    else:
        Model_before = Global_entropy.model_list[-2]
        Model_curr = Global_entropy.model_list[-1]
        before_prob = Model_before.predict_proba(train_D.x)
        curr_prob = Model_curr.predict_proba(train_D.x)
        for item in nn_list:
            item["DInf"] = abs(before_prob[item["index"]][0] - curr_prob[item["index"]][0])
            item["DNoi"] = 0
            if item["y_true"] == 1:
                item["DNoi"] = curr_prob[item["index"]][0]
            if item["y_true"] == -1:
                item["DNoi"] = curr_prob[item["index"]][1]
        nn_list = dict_normal(nn_list, "DInf")
        nn_list = dict_normal(nn_list, "SInf")
        nn_list = dict_normal(nn_list, "DNoi")
        nn_list = dict_normal(nn_list, "SNoi")
        for item in nn_list:
            item["Content_Inf"] = (1 - lamba_R) * item["SInf"] + lamba_R * item["DInf"]
            item["Content_Noi"] = (1 - lamba_R) * item["SNoi"] + lamba_R * item["DNoi"]
            item["value"] = (1 - lamba_U) * item["Content_Inf"] - lamba_U * item["Content_Noi"]


def CG_UnderSample(nn_list, train_D, Gloabl_item=None):
    Undersample_list = []
    Pro_x = []
    Pro_y = []
    for item in nn_list:
        if item["y_true"] == -1 and item['value'] < 0:
            Undersample_list.append(item)
        else:
            Pro_x.append(item["curr_item"])
            Pro_y.append(item["y_true"])
    train_D_copy = copy.deepcopy(train_D)


    if Counter(Pro_y)[-1] < Counter(Pro_y)[1] / 2:
        Undersample_list = sorted(Undersample_list, key=lambda x: x['value'], reverse=True)
        for i in range(int(Counter(Pro_y)[1])):
            Pro_x.append(Undersample_list[i]["curr_item"])
            Pro_y.append(Undersample_list[i]["y_true"])

    Pro_Data = Data_Entropy(np.array(Pro_x), np.array(Pro_y))
    return Pro_Data


def KnnAndRknnPseudo(test_D, Data_l, Global):
    test_x = test_D.x
    test_y = test_D.y
    Entropy_list = Global.Entropy_list
    res = []
    target_x = list(chain(list(test_D.x), list(Data_l.x)))
    target_y = np.array(list(chain(list(test_D.pred), list(Data_l.y))))
    target_x = np.array(target_x)
    RKNN_Matrix, RKNN_Samples = KNN_nei_Matrix_Pseudo(target_x, test_x, K=Global.K)
    for index, item in enumerate(test_x):
        RKNN_idx = RKNN_nei(RKNN_Matrix, index)
        KNN_idx = RKNN_Matrix[index]
        ls_dict = {}
        ls_dict["index"] = index
        ls_dict["y_pred"] = target_y[index]
        ls_dict["y_true"] = test_y[index]
        ls_dict["KNN"] = KNN_idx
        ls_dict["KNN_sample"] = target_y[KNN_idx]
        ls_dict["RKNN"] = RKNN_idx
        ls_dict["RKNN_sample"] = target_y[RKNN_idx]

        ls_dict["KNN_detail"] = RKNN_Samples[index]
        ls_dict["RKNN_detail"] = target_x[RKNN_idx]
        ls_dict["curr_item"] = item

        ls_dict["SInf"] = 0
        curr_count = Counter(ls_dict["KNN_sample"])
        curr_label = ls_dict["y_true"]
        curr_Entropy = Entropy_list[curr_count[curr_label]]
        ls_dict["SInf"] = curr_Entropy
        ls_dict["SNoi"] = 0
        RKNN_sam = ls_dict["RKNN_sample"]
        Rknn_len = len(RKNN_sam)

        if Rknn_len == 0:
            ls_dict["noise"] = 0
        else:
            curr_count = Counter(RKNN_sam)
            this_list = Entropy_all_MinMax(k=Rknn_len, min_num=0.0, max_num=1)
            RKNN_samm = curr_count[curr_label]
            gap = -1
            if RKNN_samm != Rknn_len:
                gap = this_list[RKNN_samm] - this_list[RKNN_samm + 1]
            else:
                gap = this_list[RKNN_samm] - 0
            nois_Entropy = this_list[RKNN_samm] - gap * RKNN_samm / (Rknn_len + 1)
            ls_dict["SNoi"] = nois_Entropy
            ls_dict["noise"] = nois_Entropy ** (1 / (Rknn_len - curr_count[curr_label] + 1))
        res.append(ls_dict)
    return res

def AddSamplesPseudo(Pseudo_Entropy, over_num):
    res_pred_pos = []
    ls_list = []
    for idx, item in enumerate(Pseudo_Entropy):
        if item["y_pred"] == 1:
            res_pred_pos.append(idx)
        else:
            continue
        if item["y_true"] == 1:
            pass
    return list(set(ls_list) | set(res_pred_pos))


def CG_OverSamples(nn_list, train_D, test_D, train_Pro, Global_item=None):
    ls_tj = Counter(train_Pro.y)
    need_over_num = ls_tj[-1] - ls_tj[1]
    if need_over_num <= 0:
        return train_Pro
    min_list = []
    Pseudo_Entropy = KnnAndRknnPseudo(test_D, train_D, Global_item)

    CG_CalPseudo(Pseudo_Entropy, test_D, Global_item)
    Pseudo_index = AddSamplesPseudo(Pseudo_Entropy, need_over_num)
    acc_num = 0
    Global_item.Pseudo_num = len(Pseudo_index)
    for item in Pseudo_index:
        curr_add_Pse = Pseudo_Entropy[item]
        if Pseudo_Entropy[item]["y_true"] == 1:
            acc_num += 1
    train_Pro.x = list(train_Pro.x)
    train_Pro.y = list(train_Pro.y)
    for idx in Pseudo_index:
        train_Pro.x.append(test_D.x[idx])
        train_Pro.y.append(1)
    return train_Pro

def OverSamples_SMOTE_ENTROPY_(train_Pro, nn_list, Global_entropy) -> Data_Entropy:
    train_Pro.source = [0] * len(train_Pro.y)
    SMOTE_NUM = Counter(train_Pro.y)[-1]
    if SMOTE_NUM <= 0:
        return train_Pro
    min_Sam_list = []
    for idx, item in enumerate(nn_list):
        if item["y_true"] == 1:
            min_Sam_list.append(item)
    trans_aim_value = []
    min_Sam_list_item = []
    for idx, item in enumerate(min_Sam_list):
        trans_aim_value.append(item["value"])
        min_Sam_list_item.append(item["curr_item"])
    Curr_K_choose = int(math.sqrt(len(min_Sam_list_item)))
    normalized_aim_value = (trans_aim_value - np.min(trans_aim_value)) / (
            np.max(trans_aim_value) - np.min(trans_aim_value))
    sum_aim_value = sum(normalized_aim_value)
    normalized_aim_value = normalized_aim_value / sum_aim_value
    if np.isnan(np.array(normalized_aim_value)).any():
        choose_from = np.random.choice(len(min_Sam_list), size=SMOTE_NUM)
    else:
        choose_from = np.random.choice(len(min_Sam_list), size=SMOTE_NUM, p=normalized_aim_value)
    choose_to = np.random.randint(low=0, high=Curr_K_choose, size=SMOTE_NUM)
    random_GAP = np.random.rand(SMOTE_NUM)
    knn = KNN(k=Curr_K_choose + 1)
    knn.fit(min_Sam_list_item)
    for idx, choose_from_item in enumerate(choose_from):
        curr_item = min_Sam_list[choose_from_item]
        Choose_FROM_ITEM = curr_item["curr_item"]
        _, samples, _ = knn.predict(Choose_FROM_ITEM)
        samples = samples[1:]
        choose_to_idx = choose_to[idx]
        Choose_TO_ITEM = samples[choose_to_idx]
        gap = Choose_TO_ITEM - Choose_FROM_ITEM
        new_Sample = Choose_FROM_ITEM + gap * random_GAP[idx]
        train_Pro.x = list(train_Pro.x)
        train_Pro.x.append(new_Sample)
        train_Pro.x = np.array(train_Pro.x)
        train_Pro.y = list(train_Pro.y)
        train_Pro.y.append(1)
        train_Pro.y = np.array(train_Pro.y)
        train_Pro.source.append(1)
    return train_Pro


def remove_syn_min(train_Pro, Global_entropy):
    x = train_Pro.x
    x = np.array(x)
    y = train_Pro.y
    y = np.array(y)
    source = train_Pro.source
    if Counter(source)[1] == 0:
        return train_Pro
    Syn_list = []
    Entropy_list = Global_entropy.Entropy_list
    K = Global_entropy.K
    KNN_Matrix, KNN_SAMPLES = KNN_nei_Matrix(x, K)
    Syn_numpy_list = []
    for index, item in enumerate(x):
        if source[index] == 0:
            continue
        Syn_numpy_list.append(item)
        RKNN_idx = RKNN_nei(KNN_Matrix, index)
        KNN_idx = KNN_Matrix[index]
        ls_dict = {}
        ls_dict["index"] = index
        ls_dict["y_true"] = y[index]
        ls_dict["KNN"] = KNN_idx
        ls_dict["KNN_sample"] = y[KNN_idx]
        ls_dict["RKNN"] = RKNN_idx
        ls_dict["RKNN_sample"] = y[RKNN_idx]
        ls_dict["KNN_detail"] = KNN_SAMPLES[index]
        ls_dict["RKNN_detail"] = x[RKNN_idx]
        ls_dict["curr_item"] = item
        curr_label = 1
        curr_count = Counter(ls_dict["KNN_sample"])
        curr_Entropy = Entropy_list[curr_count[curr_label]]
        ls_dict["SInf"] = curr_Entropy
        Rknn_len = len(ls_dict["RKNN_sample"])
        if Rknn_len == 0:
            ls_dict["SNoi"] = 0
        else:
            curr_count = Counter(ls_dict["RKNN_sample"])
            this_list = Entropy_all_MinMax(k=Rknn_len, min_num=0.0, max_num=1)
            RKNN_samm = curr_count[curr_label]
            gap = -1
            if RKNN_samm != Rknn_len:
                gap = this_list[RKNN_samm] - this_list[RKNN_samm + 1]
            else:
                gap = this_list[RKNN_samm] - 0
            nois_Entropy = this_list[RKNN_samm] - gap * RKNN_samm / (Rknn_len + 1)

            ls_dict["SNoi"] = nois_Entropy
            ls_dict["before_noise"] = nois_Entropy ** (1 / (Rknn_len - curr_count[curr_label] + 1))
        Syn_list.append(ls_dict)
    del_res = []
    Del_NUM = 0
    lamba_R = Global_entropy.lamba_aim1
    lamba_U = Global_entropy.lamba_aim2
    No = Global_entropy.No_T
    if No == 0:
        Syn_list = dict_normal(Syn_list, "SInf")
        Syn_list = dict_normal(Syn_list, "SNoi")
        for item in Syn_list:
            item["value"] = (1 - lamba_U) * item["SInf"] - lamba_U * item["SNoi"]
            if item["value"] < 0:
                del_res.append(item["index"])
                Del_NUM += 1
    else:
        Model_before = Global_entropy.model_list[-2]
        Model_curr = Global_entropy.model_list[-1]
        for item in Syn_list:
            before_prob = Model_before.predict_proba([item["curr_item"]])[0][1]
            curr_prob = Model_curr.predict_proba([item["curr_item"]])[0][1]

            item["DInf"] = abs(before_prob - curr_prob)

            item["DNoi"] = Model_curr.predict_proba([item["curr_item"]])[0][0]

        Syn_list = dict_normal(Syn_list, "SInf")
        Syn_list = dict_normal(Syn_list, "SNoi")
        Syn_list = dict_normal(Syn_list, "DNoi")
        Syn_list = dict_normal(Syn_list, "DInf")
        for item in Syn_list:
            item["Content_Inf"] = (1 - lamba_R) * item["SInf"] + lamba_R * item["DInf"]
            item["Content_Noi"] = (1 - lamba_R) * item["SNoi"] + lamba_R * item["DNoi"]
            item["value"] = (1 - lamba_U) * item["Content_Inf"] - lamba_U * item["Content_Noi"]
            if item["value"] < 0:
                del_res.append(item["index"])
                Del_NUM += 1
    train_Pro.x = np.delete(train_Pro.x, del_res, axis=0)
    train_Pro.y = np.delete(train_Pro.y, del_res)
    train_Pro.source = np.delete(train_Pro.source, del_res)
    return train_Pro


def Is_Break(Global_entropy, train_D):
    before_model = Global_entropy.model_list[-2]
    curr_model = Global_entropy.model_list[-1]
    before_pred = before_model.predict_proba(train_D.x)[:,1]
    curr_pred = curr_model.predict_proba(train_D.x)[:,1]
    b_value = AUC(train_D.y, before_pred)
    c_value = AUC(train_D.y, curr_pred)
    if c_value == 1.0:
        return -1
    if b_value > c_value or b_value == 1.0:
        return 1
    if b_value <= c_value:
        return 0


def CHRE_ensemble(model_list_v, train_D ,test):
    test_x = test.x
    test_res = [0] * len(test_x)
    weight_model = []
    for model in model_list_v:
        train_pred = model.predict_proba(train_D.x)[:,1]
        weight_curr = AUC(train_D.y, train_pred)
        weight_model.append(weight_curr)
        test_pred = model.predict(test_x)
        test_res += weight_curr * test_pred
    test_res = np.array(test_res)
    res = np.sign(test_res)
    return res




