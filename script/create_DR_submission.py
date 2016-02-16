import numpy as np
import pandas as pd
import optcut as oc
import optoffset as oo
import os
ROOT = os.path.abspath(os.path.dirname(__file__))
SUBMISSION = ROOT.replace("script", "tmp/submissions/")


if __name__ == "__main__":
    train_df = pd.read_csv(SUBMISSION + "DR_train.csv")
    test_df = pd.read_csv(SUBMISSION + "DR_test.csv")
    train_ho = train_df[train_df["Partition"] == "Holdout"]
    train_score = train_ho["Cross-Validation Prediction"].values[None].T
    train_label = train_ho["target"].values
    oc_obj = oc.OptimCutPoint()
    oo_obj = oo.OptimOffset()
    oc_obj.fit(train_score, train_label)
    oo_obj.fit(train_score, train_label)
    test_score = test_df["Prediction"].values[None].T
    test_predict = oc_obj.transform(test_score)
    test_df["Response"] = test_predict
    test_df["Id"] = test_df["ID"]
    test_df[["Id", "Response"]].to_csv(SUBMISSION + "submission_DR_output.csv",
                                       index=False)
    test_predict = oc_obj.transform(test_score)
    test_df["Response"] = test_predict
    test_df["Id"] = test_df["ID"]
    test_df[["Id", "Response"]].to_csv(SUBMISSION + "submission_DR_offset_output.csv",
                                       index=False)