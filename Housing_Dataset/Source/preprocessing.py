import numpy as np
import pandas as pd
import scipy.stats as st
from collections import Counter
from sklearn.decomposition import PCA

# 1. Load and merge data

zoning_df = pd.read_csv('data/zoning.csv')
listing_df = pd.read_csv('data/listing.csv')
sale_df = pd.read_csv('data/sale.csv')

housing_df = pd.merge(zoning_df, listing_df, left_on="Id", right_on="Id")
housing_df = pd.merge(housing_df, sale_df, left_on="Id", right_on="Id")

housing_df.set_index("Id", inplace=True)

# 2. Assign correct data types

for column in housing_df.select_dtypes(['object']).columns:
    housing_df[column] = housing_df[column].astype('category')

housing_df.MSSubClass = housing_df.MSSubClass.astype('category')
housing_df.OverallQual = housing_df.OverallQual.astype('category')
housing_df.OverallCond = housing_df.OverallCond.astype('category')
housing_df.BsmtFullBath = housing_df.BsmtFullBath.astype('category')
housing_df.BsmtHalfBath = housing_df.BsmtHalfBath.astype('category')
housing_df.FullBath = housing_df.FullBath.astype('category')
housing_df.HalfBath = housing_df.HalfBath.astype('category')
housing_df.BedroomAbvGr = housing_df.BedroomAbvGr.astype('category')
housing_df.KitchenAbvGr = housing_df.KitchenAbvGr.astype('category')
housing_df.TotRmsAbvGrd = housing_df.TotRmsAbvGrd.astype('category')
housing_df.Fireplaces = housing_df.Fireplaces.astype('category')
housing_df.GarageCars = housing_df.GarageCars.astype('category')
housing_df.MoSold = housing_df.MoSold.astype('category')

# 3. Handle missing values

housing_df.LotFrontage.fillna(housing_df.LotFrontage.mean(), inplace=True)
housing_df.MasVnrArea.fillna(housing_df.MasVnrArea.mean(), inplace=True)
housing_df.GarageYrBlt.fillna(housing_df.GarageYrBlt.mean(), inplace=True)

empty_means_without = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                        "BsmtFinType2", "FireplaceQu","GarageType","GarageFinish",
                        "GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

def replace_empty(feature, value):
    housing_df[feature].cat.add_categories([value], inplace=True)
    housing_df[feature].fillna(value, inplace=True)

for feature in empty_means_without:
    replace_empty(feature, "None")

housing_df.dropna(inplace=True)

# 4. Preprocessing

numeric_df = housing_df.select_dtypes(exclude=['category']).copy()
numeric_df.drop("SalePrice", axis=1, inplace=True)

categorical_df = housing_df.select_dtypes(include=['category']).copy()
categorical_encoded_df = pd.get_dummies(categorical_df)


numeric_log_df = np.log(numeric_df + 1)

numeric_log_std_sc_df = (numeric_log_df - numeric_log_df.mean())/numeric_log_df.std()
numeric_log_gel_sc_df = (numeric_log_df - numeric_log_df.mean())/(2*numeric_log_df.std())

stats = pd.DataFrame()
stats['mean'] = categorical_encoded_df.mean()
stats['std'] = categorical_encoded_df.std()
stats['var'] = categorical_encoded_df.var()
categorical_encoded_features_significant_variance_index = stats[stats['var'] > 0.2].sort_values('std', ascending=False).index
categorical_encoded_features_insignificant_variance_index = stats[stats['var'] <= 0.2].sort_values('std', ascending=False).index
categorical_encoded_features_significant_variance = categorical_encoded_df[categorical_encoded_features_significant_variance_index]
categorical_encoded_features_insignificant_variance = categorical_encoded_df[categorical_encoded_features_insignificant_variance_index]
categorical_encoded_features_significant_variance_centered = (categorical_encoded_features_significant_variance - 
                                                              categorical_encoded_features_significant_variance.mean())

numeric_log_std_sc_df = (numeric_log_df - numeric_log_df.mean())/numeric_log_df.std()
numeric_log_gel_sc_df = (numeric_log_df - numeric_log_df.mean())/(2*numeric_log_df.std())

# outliers

def feature_outliers(dataframe, col, param=1.5):
    Q1 = np.percentile(dataframe[col], 25)
    Q3 = np.percentile(dataframe[col], 75)
    tukey_window = param*(Q3-Q1)
    less_than_Q1 = dataframe[col] < Q1 - tukey_window
    greater_than_Q3 = dataframe[col] > Q3 + tukey_window
    tukey_mask = (less_than_Q1 | greater_than_Q3)
    return dataframe[tukey_mask]

def multiple_outliers(dataframe, count=2):
    raw_outliers = []
    for col in dataframe:
        outlier_df = feature_outliers(dataframe, col)
        raw_outliers += list(outlier_df.index)

    outlier_count = Counter(raw_outliers)
    outliers = [k for k,v in outlier_count.items() if v >= count]
    return outliers

numeric_log_std_sc_out_rem_df = numeric_log_std_sc_df.drop(multiple_outliers(numeric_log_std_sc_df, 5))
categorical_encoded_out_rem_df = categorical_encoded_df.drop(multiple_outliers(numeric_log_std_sc_df, 5))

numeric_log_gel_sc_out_rem_df = numeric_log_gel_sc_df.drop(multiple_outliers(numeric_log_gel_sc_df, 5))
categorical_encoded_features_insignificant_variance_out_rem = categorical_encoded_features_insignificant_variance.drop(multiple_outliers(numeric_log_gel_sc_df, 5))
categorical_encoded_features_significant_variance_centered_out_rem = categorical_encoded_features_significant_variance_centered.drop(multiple_outliers(numeric_log_gel_sc_df, 5))
                                                               
# principal component analysis

numeric_gelman_categorical_significant = pd.merge(numeric_log_gel_sc_out_rem_df, 
                                                  categorical_encoded_features_significant_variance_centered_out_rem,
                                                  left_index=True, right_index=True)

pca_log_std_sc_out_rem = PCA(8)
pca_num_gel_cat = PCA(8)

pca_log_std_sc_out_rem.fit(numeric_log_std_sc_out_rem_df)
pca_num_gel_cat.fit(numeric_gelman_categorical_significant)