#!/usr/bin/env python
'''
@Project ：main.py 
@Author  ：zly
@Date    ：2023/11/16 10:03 
'''
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


from utils_CHRE import *

def main(lamba_aim1, lamba_aim2, K=5):
    # lamba_aim1:评估动态和静态，越大动态越高
    # lamba_aim2:评估信息与噪声，越大噪声越重视
    np.random.seed(42)
    # 设置迭代值
    iter_T = 10
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    Entropy_list = Entropy_all_MinMax(k=K)
    filename = "pima.txt"
    dir_filename = "data/{}".format(filename)
    print(filename)
    x, y = loadfile(dir_filename)
    mm = StandardScaler()
    x = mm.fit_transform(x)
    # K_fold
    all_data = Stratified_fold_K_version_2(x, y, n_spli=5)
    for No in range(1):
        split_list = []
        train_data, train_labels = all_data["train_x"][No], all_data["train_y"][No]
        test_data, test_labels = all_data["test_x"][No], all_data["test_y"][No]
        # 定义一些train,test的变量
        train_D = Data_Entropy(train_data, train_labels)
        test_D = Data_Entropy(test_data, test_labels)
        # 用于保存一些变量
        Global_entropy = cg_Global()
        # 初始数据集的结果
        model.fit(train_D.x, train_D.y)
        # model_list
        model_list_v = []
        model_list_v.append(copy.deepcopy(model))
        # 将model保存
        Global_entropy.model_list.append(copy.deepcopy(model))
        Global_entropy.Data_list.append(copy.deepcopy(train_D))
        Global_entropy.Entropy_list = Entropy_list
        Global_entropy.K = K
        Global_entropy.lamba_aim1 = lamba_aim1
        Global_entropy.lamba_aim2 = lamba_aim2
        ########### 预测测试集的标签
        # 获得预测标签
        y_pred = model.predict(test_D.x)
        # 获得初始的概率
        prob_data = model.predict_proba(test_D.x)
        test_D.pred = y_pred
        test_D.pred_prob = prob_data
        auc, f1 = AUC(test_D.y, y_pred),  f1_score(test_D.y, y_pred)
        # 初始结果
        ########### 预测训练集的标签
        train_y_pred = model.predict(train_data)
        train_D.pred = train_y_pred
        # 计算熵与置信度相加的公式
        # 获取KNN和RKNN的list
        # step 1:计算静态的数据邻域以及影响邻域
        nn_list = KnnAndRknn(train_D, K=K, Gloabl_item=Global_entropy)
        No_T = 0
        best_model = None
        best_Data = None
        while No_T < iter_T:
            print('当前迭代为:', No_T)
            Global_entropy.No_T = No_T
            # 根据公式9计算Value值
            cal_target_value(nn_list, train_D, Global_entropy)
            # 获得需要删除的多数类下标
            Pro_Data = CG_UnderSample(nn_list, train_D, Gloabl_item=Global_entropy)
            print("欠采样:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            model_list_v.append(copy.deepcopy(model))
            # 进行过采样
            Pro_Data = OverSamples_SMOTE_ENTROPY_1016(Pro_Data, nn_list, Global_entropy)
            print("过采样:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            model_list_v.append(copy.deepcopy(model))
            remove_syn_min(Pro_Data, Global_entropy)
            # 后处理后
            print("后处理:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            model_list_v.append(copy.deepcopy(model))
            No_T += 1
            # 保存当前的模型
            Global_entropy.model_list.append(copy.deepcopy(model))
            Global_entropy.Data_list.append(copy.deepcopy(Pro_Data))
            # 这边判断调出条件
            break_is = Is_Break(Global_entropy, train_D)
            if break_is == 1:
                best_model = Global_entropy.model_list[-2]
                best_Data = Global_entropy.Data_list[-2]
                print("第{}调出迭代".format(No_T))
                break
            if break_is == 0:
                best_model = Global_entropy.model_list[-1]
                best_Data = Global_entropy.Data_list[-1]
            if break_is == -1:
                best_model = Global_entropy.model_list[-1]
                best_Data = Global_entropy.Data_list[-1]
                print("第{}调出迭代".format(No_T))
                break
        print("最终:")
        y_pred = CHRE_ensemble(model_list_v, train_D ,test_D)
        auc, f1 = AUC(test_D.y, y_pred),  f1_score(test_D.y, y_pred)
        print(auc, f1)

if __name__ == '__main__':
    main(0.5,0.5)