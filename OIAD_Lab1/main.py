# import sys
import pandas as pd
import mglearn
# import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
# import scipy as sp
# import IPython
# import sklearn

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("\n\n=========================  Ключи iris_dataset  =========================\n{}".format(iris_dataset.keys()))
# print(iris_dataset['DESCR'][:1100] + "\n...")
print("\n\n=====================  Названия ответов (target)  ======================\n\t\t\t\t  {}".format(iris_dataset['target_names']))
print("\n\n====================  Названия признаков (feature)  ====================\n{}".format(iris_dataset['feature_names']))

print("\n\n=============   Тип массива data: {}".format(type(iris_dataset['data'])))
print("=============\t\t  Форма массива data: {}".format(iris_dataset['data'].shape))
print("======\tПервые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))

print("\n\n=============   Тип массива target: {}".format(type(iris_dataset['target'])))
print("=============\t\t  Форма массива target: {}".format(iris_dataset['target'].shape))
print("======\tОтветы:\n{}".format(iris_dataset['target'][:150]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
print("\n")
print("======\tФорма массива X_train: {}".format(X_train.shape))
print("======\tФорма массива y_train: {}".format(y_train.shape))
print("======\tФорма массива X_test:  {}".format(X_test.shape))
print("======\tФорма массива y_test:  {}".format(y_test.shape))

# создаем dataframe изданныхвмассиве X_train
# маркируемстолбцы, используястрокив iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# создаемматрицурассеянияиз dataframe, цветточекзадаемспомощью y_train
from pandas.plotting import scatter_matrix
grr = scatter_matrix(iris_dataframe,c=y_train,figsize=(15, 15),marker='o',hist_kwds={'bins': 20},s=60,alpha=.8,cmap=mglearn.cm3)
plt.show()
