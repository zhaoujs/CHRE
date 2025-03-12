
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from utils import *


def main(filename, lamba_aim1, lamba_aim2, K=5):
    iter_T = 10
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    Entropy_list = Entropy_all_MinMax(k=K)
    dir_filename = "../data/{}".format(filename)
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
        test_D = Data_Entropy(test_data, test_labels)

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
            Global_entropy.No_T = No_T
            cal_target_value(nn_list, train_D, Global_entropy)
            Pro_Data = CG_UnderSample(nn_list, train_D, Gloabl_item=Global_entropy)
            Pro_Data = OverSamples_SMOTE_ENTROPY_(Pro_Data, nn_list, Global_entropy)
            remove_syn_min(Pro_Data, Global_entropy)
            No_T += 1
            Global_entropy.model_list.append(copy.deepcopy(model))
            Global_entropy.Data_list.append(copy.deepcopy(Pro_Data))
            break_is = Is_Break(Global_entropy, train_D)
            if No_T < 2:
                break_is = 0
            if break_is == 1:
                break
            if break_is == 0:
                pass
            if break_is == -1:
                break

        y_pred = CHRE_ensemble(
            Global_entropy.model_list,
            train_D, test_D)


if __name__ == '__main__':
    filename = "pima.txt"
    main(filename, 0.5, 0.5)
