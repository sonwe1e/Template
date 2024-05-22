# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def main():
    data = []
    with open("./data/finvcup9th_1st_ds5/train_label.txt", "r") as f:
        for tmp in f.readlines():
            data.append(tmp.replace("\n", "").split(","))
    df_data = pd.DataFrame(data, columns=["wav_path", "label"])
    df_data["wav_path"] = df_data["wav_path"].apply(
        lambda x: "./data/finvcup9th_1st_ds5/train/" + x
    )

    # n 折划分
    n_splits = 6  # 设置 n 值
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    paths = df_data["wav_path"].values
    labels = df_data["label"].values

    for fold, (train_index, valid_index) in enumerate(kf.split(paths)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_valid = paths[train_index], paths[valid_index]
        y_train, y_valid = labels[train_index], labels[valid_index]

        df_train = pd.DataFrame(X_train, columns=["wav_path"])
        df_train["label"] = y_train

        df_valid = pd.DataFrame(X_valid, columns=["wav_path"])
        df_valid["label"] = y_valid

        df_train.to_csv(
            f"./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_train_data_fold_{fold + 1}.csv",
            index=False,
        )
        df_valid.to_csv(
            f"./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_valid_data_fold_{fold + 1}.csv",
            index=False,
        )

    # 测试集生成
    test_speeches = glob.glob(os.path.join("./data/finvcup9th_1st_ds5/test", "*.wav"))
    df_test = pd.DataFrame(test_speeches, columns=["wav_path"])
    df_test.to_csv(
        "./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_test_data.csv", index=False
    )

    print("done!")


if __name__ == "__main__":
    main()
