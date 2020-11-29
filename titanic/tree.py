# https://qiita.com/suzuki-navi/items/d0e33a5379b4bfb848bc 
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pydotplus
from IPython.display import Image
#from sklearn.externals.six import StringIO
from six import StringIO


# read csv data
train_path = "./dat/train.csv"
test_path = "./dat/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# make table of missing percentage
def kesson_table(df):
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : 'Defects', 1 : '%'})
        return kesson_table_ren_columns

# check
print(kesson_table(train))
print(kesson_table(test))

# provide missing data of Age and Embarked with median
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# check
print(kesson_table(train))
print(kesson_table(test))


# change word to number
train = pd.concat([train, pd.get_dummies(train["Embarked"], prefix="Embarked")], axis=1).drop(columns=["Embarked"])
print(train)
train["Sex"] = pd.get_dummies(train["Sex"], drop_first=True)

test = pd.concat([test, pd.get_dummies(test["Embarked"], prefix="Embarked")], axis=1).drop(columns=["Embarked"])
test["Sex"] = pd.get_dummies(test["Sex"], drop_first=True)


# check
print(train.head(10))
print(test.head(10))


# get target and feature data
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# make tree
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# visual tree
#dot_data = StringIO()
#export_graphviz(my_tree_one, out_file='tree.dot',
#                     feature_names=["Pclass", "Sex", "Age", "Fare"],
#                     class_names=["False","True"],
#                     filled=True, rounded=True,
#                     special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())

