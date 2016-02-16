# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, KFold
from ml_metrics import quadratic_weighted_kappa
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import multiprocessing as mp
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
import features
import os
import sys
import optcut as oc
import optoffset as oo
clf_dict = {
    'Ridge': {
        "name": 'Ridge Regression',
        "clf": sklearn.linear_model.Ridge,
        "paramteters": {'alpha': [0.1, 1.0]}
    },
    'GB': {
        "name": 'Gradient Boosting',
        "clf": sklearn.ensemble.GradientBoostingRegressor,
        "paramteters": {
            'learning_rate': [0.005],
            'n_estimators': [100, 250],
            'max_depth': [4, 7],
            'random_state': [1]
        }
    },
    'RF': {
        "name": 'Random Forest',
        "clf": sklearn.ensemble.RandomForestRegressor,
        "paramteters": {
            'n_estimators': [250, 500],
            'max_features': [0.6],
            'max_leaf_nodes': [1000, 2000],
            'random_state': [1]
        }
    },
    'SVR': {
        "name": 'SVR',
        "clf": sklearn.svm.SVR,
        "paramteters": {
            'C': [1.0, 10.0]
        }
    }
}
try:
    import xgboost as xgb
    clf_dict["XGB_REG"] = {
        "name": "XGBoost",
        "clf": xgb.XGBRegressor,
        "paramteters": {
            "n_estimators": [1500],
            "max_depth": [6, 8],
            "subsample": [0.7, 0.9],
            "objective": ["reg:linear"],
            "learning_rate": [0.025],
            "colsample_bytree": [0.67, 1],
            "min_child_weight": [1.0, 100, 250],
            "seed": [10]
        }
    }

    clf_dict["XGB_REG2"] = {
        "name": "XGBoost2",
        "clf": xgb.XGBRegressor,
        "paramteters": {
            "n_estimators": [1500],
            "max_depth": [5, 7],
            "objective": ["reg:linear"],
            "learning_rate": [0.025],
            "colsample_bytree": [0.67, 1],
            "subsample": [0.7, 0.9],
            "min_child_weight": [1.0, 100, 250],
            "seed": [1]
        }
    }

    clf_dict["XGB_RANK"] = {
        "name": "XGBoost_Rank",
        "clf": xgb.XGBRegressor,
        "paramteters": {
            "n_estimators": [1500, 2500],
            "max_depth": [6, 8, 10],
            "objective": ["rank:pairwise"]
        }
    }
except ImportError:
    pass

ROOT = os.path.abspath(os.path.dirname(__file__))
CSVDIR = ROOT.replace("script", "tmp/csvdata/")
SUBMISSION = ROOT.replace("script", "tmp/submissions/")

LOG_DICT = {
    "C": [1, 10],
    "class_weight": ["balanced", None]
}


def transform(self, X):

    return self.predict(X)[None].T


def load_train():
    train_csv = CSVDIR + "train.csv"
    df = pd.read_csv(train_csv)

    return df


def load_test():
    test_csv = CSVDIR + "test.csv"
    df = pd.read_csv(test_csv)

    return df


def get_split_feature_list(fu):
    """
    :param sklearn.pipeline.FeatureUnion fu:
    :rtype: list.
    """

    return [a.split("__")[1] for a in fu.get_feature_names()]


def output_function(x):
    if x<1:

        return 1
    elif x>8:

        return 8
    else:

        return int(round(x))


def qwk_score(y_true, y_pred):
    kappa = quadratic_weighted_kappa(y_true, y_pred)

    return kappa

qwk = make_scorer(qwk_score, greater_is_better=True)


def prediction(train_df, test_df, MODELS):
    print("...create feature")
    fu_obj = FeatureUnion(transformer_list=features.get_feature_list())
    train_X = fu_obj.fit_transform(train_df, train_df["Response"])
    train_y = train_df["Response"].as_matrix()
    train_dump_df = pd.DataFrame(train_X, columns=get_split_feature_list(fu_obj))
    train_dump_df["target"] = train_y
    train_dump_df["ID"] = -1
    train_dump_df.to_csv(SUBMISSION + "train_dump.csv", index=False)
    test_X = fu_obj.transform(test_df)
    test_dump_df = pd.DataFrame(test_X, columns=get_split_feature_list(fu_obj))
    test_dump_df["ID"] = test_df["Id"]
    test_dump_df.to_csv(SUBMISSION + "test_dump.csv", index=False)
    oc_obj = oc.OptimCutPoint()
    oc_obj2 = oc.OptimCutPoint()
    oo_obj = oo.OptimOffset()
    oo_obj2 = oo.OptimOffset()
    oo_all_obj = oo.OptimOffset(True)
    oo_all_obj2 = oo.OptimOffset(True)
    model_list = MODELS.split(",")
    clf_list = []
    valid_list = []
    kf = KFold(len(train_X), random_state=7777, n_folds=5)
    print("...start fitting")
    for model in model_list:
        print("... start fit %s model" % model)
        valid_pred_list = []
        valid_label_list = []
        for train_index, test_index in kf:
            use_train_X = train_X[train_index]
            use_train_y = train_y[train_index]
            valid_X = train_X[test_index]
            valid_y = train_y[test_index]
            clf_dict[model]["paramteters"]
            if model == "XGB_REG" or model == "XGB_RANK" or model == "XGB_REG2":
                use_train_X, xgb_valid_X, use_train_y, xgb_valid_y =\
                    train_test_split(use_train_X, use_train_y, test_size=0.2)
                fit_param = {"eval_set": [(use_train_X, use_train_y),
                                          (xgb_valid_X, xgb_valid_y)],
                             "early_stopping_rounds": 50
                            }
                f_clf = GridSearchCV(estimator=clf_dict[model]["clf"](),
                                     param_grid=clf_dict[model]["paramteters"],
                                     n_jobs=3, verbose=2, fit_params=fit_param)
            else:
                fit_param = {}
                f_clf = GridSearchCV(estimator=clf_dict[model]["clf"](),
                                     param_grid=clf_dict[model]["paramteters"],
                                     n_jobs=3, verbose=2)
            f_clf.fit(use_train_X, use_train_y)
            valid_pred_list.append(f_clf.predict(valid_X))
            valid_label_list.append(valid_y)
        valid_list.append(np.concatenate(valid_pred_list))
        concat_valid_y = (np.concatenate(valid_label_list))
        use_train_X = np.copy(train_X)
        use_train_y = np.copy(train_y)
        if model == "XGB_REG" or model == "XGB_RANK" or model == "XGB_REG2":
            use_train_X, xgb_valid_X, use_train_y, xgb_valid_y =\
                train_test_split(train_X, train_y, test_size=0.2)
            fit_param = {"eval_set": [(use_train_X, use_train_y),
                                      (xgb_valid_X, xgb_valid_y)],
                         "early_stopping_rounds": 50
                        }
            clf = GridSearchCV(estimator=clf_dict[model]["clf"](),
                               param_grid=clf_dict[model]["paramteters"],
                               n_jobs=3, verbose=1, fit_params=fit_param)
        else:
            fit_param = {}
            clf = GridSearchCV(estimator=clf_dict[model]["clf"](),
                               param_grid=clf_dict[model]["paramteters"],
                               n_jobs=3, verbose=2)
        clf.fit(use_train_X, use_train_y)
        clf_list.append(clf)
    print("... start optim cutting")
    if len(clf_list) > 1:
        test_predict_list = [c.predict(test_X) for c in clf_list]
        valid_predict_X = np.c_[valid_list].T
        test_predict_X = np.c_[test_predict_list].T
        linear_reg = sklearn.linear_model.LinearRegression()
        linear_reg.fit(valid_predict_X, concat_valid_y)
        print(linear_reg.intercept_)
        print(linear_reg.coef_)
        valid_ave_predict = valid_predict_X.mean(axis=1)[None].T
        valid_predict = linear_reg.predict(valid_predict_X)[None].T
        test_ave_predict = test_predict_X.mean(axis=1)[None].T
        test_predict = linear_reg.predict(test_predict_X)[None].T
    else:
        use_clf = clf_list[0]
        concat_valid_y = train_y
        valid_predict = use_clf.predict(train_X)[None].T
        test_predict = use_clf.predict(test_X)[None].T
    print("...start y_pred")
    oo_obj.fit(valid_predict, concat_valid_y)
    oo_all_obj.fit(valid_predict, concat_valid_y)
    oc_obj.fit(valid_predict, concat_valid_y)
    y_pred = oo_obj.transform(test_predict)
    pred_sr = pd.Series(y_pred, name="Response", index=test_df["Id"])
    submissionfile = SUBMISSION + "submission_offset_%s.csv" % MODELS
    pred_sr.to_csv(submissionfile, header=True, index_label="ID")
    y_pred = oo_all_obj.transform(test_predict)
    pred_sr = pd.Series(y_pred, name="Response", index=test_df["Id"])
    submissionfile = SUBMISSION + "submission_offset_all_%s.csv" % MODELS
    pred_sr.to_csv(submissionfile, header=True, index_label="ID")
    y_pred = oc_obj.transform(test_predict)
    pred_sr = pd.Series(y_pred, name="Response", index=test_df["Id"])
    submissionfile = SUBMISSION + "submission_cutpoint_%s.csv" % MODELS
    pred_sr.to_csv(submissionfile, header=True, index_label="ID")
    if len(clf_list) > 1:
        oo_obj2.fit(valid_ave_predict, concat_valid_y)
        oo_all_obj2.fit(valid_predict, concat_valid_y)
        y_pred2 = oo_obj2.transform(test_ave_predict)
        pred_sr2 = pd.Series(y_pred2, name="Response", index=test_df["Id"])
        submissionfile = SUBMISSION + "submission_offset_ave_%s.csv" % MODELS
        pred_sr2.to_csv(submissionfile, header=True, index_label="ID")
        y_pred2 = oo_all_obj2.transform(test_ave_predict)
        pred_sr2 = pd.Series(y_pred2, name="Response", index=test_df["Id"])
        submissionfile = SUBMISSION + "submission_offset_all_ave_%s.csv" % MODELS
        pred_sr2.to_csv(submissionfile, header=True, index_label="ID")
        oc_obj2.fit(valid_ave_predict, concat_valid_y)
        y_pred2 = oc_obj2.transform(test_ave_predict)
        pred_sr2 = pd.Series(y_pred2, name="Response", index=test_df["Id"])
        submissionfile = SUBMISSION + "submission_cutpoint_ave_%s.csv" % MODELS
        pred_sr2.to_csv(submissionfile, header=True, index_label="ID")
    print("... finish y_pred")


if __name__ == "__main__":
    param = sys.argv
    os.environ["JOBLIB_START_METHOD"] = "forkserver"
    print(param)
    print("... load data")
    train_df = load_train()
    test_df = load_test()
    if len(param) == 1:
        print("... predict")
        prediction(train_df, test_df, "Ridge")
    else:
        method = param[1]
        if len(param) == 3:
            use_model = param[2]
        else:
            use_model = "Ridge"
        print("... predict")
        prediction(train_df, test_df, use_model)