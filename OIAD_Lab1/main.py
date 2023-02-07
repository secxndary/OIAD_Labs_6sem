import sys
import pandas as pds
# import mglearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn

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