# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


class CategorizeFeature(BaseEstimator, TransformerMixin):
    def __init__(self, c_name, threshold=0.001, target_mean=0.25,
                 fill_mean=-1):
        self._target_mean_sr = None
        self._vec = DictVectorizer(sparse=False)
        self._threshold = threshold
        self._c_name = c_name
        self._target_mean = target_mean
        self._fill_mean = fill_mean

    def get_feature_names(self):
        res = self._vec.get_feature_names()
        if self._target_mean:
            res = res + ["%s_mean" % self._c_name]

        return res

    def get_data_array(self, df):

        return df[self._c_name] \
            .apply(lambda x: {self._c_name: str(x)}).values

    def fit(self, df, y=None):
        """
        fit for categorize feature

        :param pandas.DataFrame df:
        :param pandas.Series y:
        :rtype: numpy.array
        """
        data_arr = self.get_data_array(df)
        tmp_vec = self._vec.fit_transform(data_arr)
        flg_arr = (tmp_vec.mean(axis=0) > self._threshold) &\
                  (tmp_vec.mean(axis=0) < 1 - self._threshold)
        self._vec.restrict(flg_arr)
        self._target_mean = flg_arr.mean() < (1 - self._target_mean)
        if self._target_mean:
            clone_df = df.copy()
            clone_df["target"] = y
            self._target_mean_sr = clone_df.groupby(self._c_name)["target"].mean()

        return self

    def get_target_mean(self, sr):
        target_index = sr[self._c_name]
        if target_index in self._target_mean_sr:

            return self._target_mean_sr[target_index]

        return self._fill_mean

    def transform(self, df):
        data_arr = self.get_data_array(df)
        res = self._vec.transform(data_arr)
        if self._target_mean:
            res = np.c_[res, df.apply(self.get_target_mean, axis=1).values]

        return res


class NumericFeature(BaseEstimator, TransformerMixin):
    COLUMN_NAME = None
    FEATURE_NAME = None
    DEFAULT_VAL = 0

    def __init__(self, c_name=None, d_val=None):
        if c_name is not None:
            self.COLUMN_NAME = c_name
            self.FEATURE_NAME = c_name
        if d_val is not None:
            self.DEFAULT_VAL = d_val

    def get_feature_names(self):
        return [self.FEATURE_NAME]

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df[self.COLUMN_NAME].fillna(self.DEFAULT_VAL).as_matrix()[None].T


class ProductInfo2(BaseEstimator, TransformerMixin):
    COLUMN_NAME = "Product_Info_2"
    FEATURE_NAME = "Product_Info_2"

    def __init__(self,):
        self._vec1 = DictVectorizer(sparse=False)
        self._vec2 = DictVectorizer(sparse=False)

    def get_feature_names(self):

        return self._vec1.get_feature_names() + self._vec2.get_feature_names()

    def get_data_array_1(self, df):

        return df[self.COLUMN_NAME] \
            .apply(lambda x: {self.FEATURE_NAME: str(x)[0]}).values

    def get_data_array_2(self, df):

        return df[self.COLUMN_NAME] \
            .apply(lambda x: {self.FEATURE_NAME: str(x)[1]}).values

    def fit(self, df, y=None):
        data_arr_1 = self.get_data_array_1(df)
        self._vec1.fit(data_arr_1)
        data_arr_2 = self.get_data_array_2(df)
        self._vec2.fit(data_arr_2)

        return self

    def transform(self, df):
        data_arr_1 = self.get_data_array_1(df)
        data_arr_2 = self.get_data_array_2(df)
        res_1 = self._vec1.transform(data_arr_1)
        res_2 = self._vec2.transform(data_arr_2)

        return np.c_[res_1, res_2]


class MedicalKeywordTfIdf(BaseEstimator, TransformerMixin):
    MEDICAL_KEYWORDS = ["Medical_Keyword_" + str(i) for i in range(1, 49)]

    def __init__(self):
        self._vec = TfidfVectorizer(max_df=0.95, min_df=2)

    def get_feature_names(self):

        return [x + "_TFIDF" for x in self._vec.get_feature_names()]

    def get_data_array(self, df):

        return df[self.MEDICAL_KEYWORDS] \
            .apply(lambda x: " ".join(x[x == 1].index), axis=1).values

    def fit(self, df, y=None):
        data_arr = self.get_data_array(df)
        self._vec.fit(data_arr)

        return self

    def transform(self, df):
        data_arr = self.get_data_array(df)

        return self._vec.transform(data_arr).toarray()


class MedicalKeywordCount(BaseEstimator, TransformerMixin):
    MEDICAL_KEYWORDS = ["Medical_Keyword_" + str(i) for i in range(1, 49)]

    @staticmethod
    def get_feature_names():

        return ["Medical_Keyword_Count"]

    def fit(self, df, y=None):

        return self

    def transform(self, df):

        return df[self.MEDICAL_KEYWORDS].sum(axis=1).values[None].T


class BMI_AGE(BaseEstimator, TransformerMixin):

    @staticmethod
    def get_feature_names():

        return ["BMI_AGE"]

    def fit(self, df, y=None):

        return self

    def transform(self, df):

        return (df["BMI"] * df["Ins_Age"]).values[None].T


def get_feature_list():
    feature_list = []
    categorical_columns = [
        "Product_Info_1",
        # "Product_Info_2",
        "Product_Info_3",
        "Product_Info_5",
        "Product_Info_6",
        "Product_Info_7",
        "Employment_Info_2",
        "Employment_Info_3",
        "Employment_Info_5",
        "InsuredInfo_1",
        "InsuredInfo_2",
        "InsuredInfo_3",
        "InsuredInfo_4",
        "InsuredInfo_5",
        "InsuredInfo_6",
        "InsuredInfo_7",
        "Insurance_History_1",
        "Insurance_History_2",
        "Insurance_History_3",
        "Insurance_History_4",
        "Insurance_History_7",
        "Insurance_History_8",
        "Insurance_History_9",
        "Family_Hist_1",
        "Medical_History_2",
        "Medical_History_3",
        "Medical_History_4",
        "Medical_History_5",
        "Medical_History_6",
        "Medical_History_7",
        "Medical_History_8",
        "Medical_History_9",
        "Medical_History_11",
        "Medical_History_12",
        "Medical_History_13",
        "Medical_History_14",
        "Medical_History_16",
        "Medical_History_17",
        "Medical_History_18",
        "Medical_History_19",
        "Medical_History_20",
        "Medical_History_21",
        "Medical_History_22",
        "Medical_History_23",
        "Medical_History_25",
        "Medical_History_26",
        "Medical_History_27",
        "Medical_History_28",
        "Medical_History_29",
        "Medical_History_30",
        "Medical_History_31",
        "Medical_History_33",
        "Medical_History_34",
        "Medical_History_35",
        "Medical_History_36",
        "Medical_History_37",
        "Medical_History_38",
        "Medical_History_39",
        "Medical_History_40",
        "Medical_History_41"
    ]
    feature_list.append(("ProductInfo_2", CategorizeFeature(c_name="Product_Info_2")))

    for col in categorical_columns:
        add_fu = (col, NumericFeature(c_name=col, d_val=-1))
        feature_list.append(add_fu)

    numeric_columns = [
        "Product_Info_4",
        "Ins_Age",
        "Ht",
        "Wt",
        "BMI",
        "Employment_Info_1",
        "Employment_Info_4",
        "Employment_Info_6",
        "Insurance_History_5",
        "Family_Hist_2",
        "Family_Hist_3",
        "Family_Hist_4",
        "Family_Hist_5",
        "Medical_History_1",
        "Medical_History_10",
        "Medical_History_15",
        "Medical_History_24",
        "Medical_History_32"
    ]
    medical_keywords = ["Medical_Keyword_" + str(i) for i in range(1, 49)]

    numeric_columns += medical_keywords

    for col in numeric_columns:
        add_fu = (col, NumericFeature(c_name=col, d_val=-1))
        feature_list.append(add_fu)

    feature_list.append(("Product_Info_2", ProductInfo2()))
    feature_list.append(("Medical_Keyword_Tfidf", MedicalKeywordTfIdf()))
    feature_list.append(("Medical_Keyword_Count", MedicalKeywordCount()))
    feature_list.append(("BMI_Age", BMI_AGE()))

    return feature_list
