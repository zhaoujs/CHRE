
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from utils import *


def main(filename, lamba_aim1, lamba_aim2, K=5):
    np.random.seed(42)
    iter_T = 10
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    Entropy_list = Entropy_all_MinMax(k=K)

    dir_filename = "data/{}".format(filename)
    print(filename)
    x, y = loadfile(dir_filename)
    mm = StandardScaler()
    x = mm.fit_transform(x)
    # K_fold
    all_data = Stratified_fold_K_version_2(x, y, n_spli=5)
    filename_avg = []

    for No in range(5):
        split_list = []
        train_data, train_labels = all_data["train_x"][No], all_data["train_y"][No]
        test_data, test_labels = all_data["test_x"][No], all_data["test_y"][No]
        train_D = Data_Entropy(train_data, train_labels)
        print("original:", Counter(train_D.y))
        test_D = Data_Entropy(test_data, test_labels)
        # 用于保存一些变量
        Global_entropy = cg_Global()
        model.fit(train_D.x, train_D.y)
        # model save
        Global_entropy.model_list.append(copy.deepcopy(model))
        Global_entropy.Data_list.append(copy.deepcopy(train_D))
        Global_entropy.Entropy_list = Entropy_list
        Global_entropy.K = K
        Global_entropy.lamba_aim1 = lamba_aim1
        Global_entropy.lamba_aim2 = lamba_aim2
        y_pred = model.predict(test_D.x)
        prob_data = model.predict_proba(test_D.x)
        test_D.pred = y_pred
        test_D.pred_prob = prob_data
        train_y_pred = model.predict(train_data)
        train_D.pred = train_y_pred
        # step 1
        nn_list = KnnAndRknn(train_D, K=K, Gloabl_item=Global_entropy)
        No_T = 0
        while No_T < iter_T:
            print('iters:', No_T)
            Global_entropy.No_T = No_T
            # equal 9
            cal_target_value(nn_list, train_D, Global_entropy)
            Pro_Data = CG_UnderSample(nn_list, train_D, Gloabl_item=Global_entropy)
            print("under-sampling:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            y_pred = model.predict(test_D.x)
            auc, f1 = AUC(test_D.y, y_pred), f1_score(test_D.y, y_pred)
            print(auc, f1)
            ################## PSE
            Pro_Data = CG_OverSamples(nn_list, train_D, test_D, Pro_Data, Global_item=Global_entropy)
            print("Pseudo:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            y_pred = model.predict(test_D.x)
            auc, f1 = AUC(test_D.y, y_pred), f1_score(test_D.y, y_pred),
            print(auc, f1)
            Pro_Data = OverSamples_SMOTE_ENTROPY_(Pro_Data, nn_list, Global_entropy)
            print("over-sampling:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            y_pred = model.predict(test_D.x)
            auc, f1 = AUC(test_D.y, y_pred), f1_score(test_D.y, y_pred),
            print(auc, f1)
            remove_syn_min(Pro_Data, Global_entropy)
            # post
            print("post:", Counter(Pro_Data.y))
            model.fit(Pro_Data.x, Pro_Data.y)
            y_pred = model.predict(test_D.x)
            auc, f1 = AUC(test_D.y, y_pred), f1_score(test_D.y, y_pred),
            print(auc, f1)
            No_T += 1
            Global_entropy.model_list.append(copy.deepcopy(model))
            Global_entropy.Data_list.append(copy.deepcopy(Pro_Data))
            break_is = Is_Break(Global_entropy, train_D)
            if No_T < 2:
                break_is = 0
            if break_is == 1:
                print("No.{} exit!".format(No_T))
                break
            if break_is == 0:
                pass
            if break_is == -1:
                print("No.{} exit!".format(No_T))
                break

        print("ensemble:")
        y_pred = CHRE_ensemble(
            Global_entropy.model_list,
            train_D, test_D)
        auc, f1 = AUC(test_D.y, y_pred), f1_score(test_D.y, y_pred)
        print(auc, f1)
        split_list.extend([auc, f1])
        filename_avg.append(split_list)
    res_avg = np.mean(filename_avg, 0)
    print(res_avg)


if __name__ == '__main__':
    filename = ""
    main(filename, 0.5, 0.5)
