from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

#1 load dataset
housing = pd.read_csv("housing.csv")

#2 create stratified test set
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

# we will work on the copy of training data
housing = strat_train_set.copy()

#3 seperate feautures and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

print(housing, housing_labels)

#4 numerical and categorical columns
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

#5 lets make pipelines
#for numerical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

#for categorical attributes
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

#contruct full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

#6 transform data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

#7 train the model

#linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                     scoring='neg_root_mean_squared_error', cv=10)
#print(f'The root mean square error for linear regression is {lin_rmse}')
print(pd.Series(lin_rmses).describe())

#decision tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = cross_val_score(dec_reg, housing_prepared, housing_labels,
                                     scoring='neg_root_mean_squared_error', cv=10)
#print(f'The root mean square error for decision tree is {dec_rmses}')
print(pd.Series(dec_rmses).describe())

#random forest model
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_preds = forest_reg.predict(housing_prepared)
#forest_rmse = root_mean_squared_error(housing_labels, forest_preds)
forest_rmses = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                     scoring='neg_root_mean_squared_error', cv=10)
#print(f'The root mean square error for random forest is {forest_rmse}')
print(pd.Series(forest_rmses).describe())