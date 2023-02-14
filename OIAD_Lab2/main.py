import mglearn
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier


iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("======\tФорма массива X_train: {}".format(X_train.shape))
print("======\tФорма массива y_train: {}".format(y_train.shape))
print("======\tФорма массива X_test:  {}".format(X_test.shape))
print("======\tФорма массива y_test:  {}".format(y_test.shape))

iris_dataframe = pds.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

knn = KNeighborsClassifier(algorithm='auto',
                           leaf_size=30,
                           metric='minkowski',
                           metric_params=None,
                           n_jobs=1,
                           n_neighbors=1,
                           p=2,
                           weights='uniform')   # knn — обучающий набор, алгоритмы прогноза и построения модели
knn.fit(X_train, y_train)                       # fit() – построение модели на обучающем наборе

X_new = np.array([[5, 2.9, 1, 0.2]])            # наши новые данные в двумерном массиве, по которым надо сделать прогноз
print("======\tФорма массива X_new:   {}".format(X_new. shape))


prediction = knn.predict(X_new)                 # совершение прогноза для нового ириса
print("\n==========\tПрогноз: {}".format(prediction))
print("==========\tМетка:   {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)                    # прогноз на старом тестовом наборе
print("\n======\tПрогнозы для тестового набора:\n {}".format(y_pred))


print()
print("| Правильность через y_pred == y_test:  {:.2f}".format(np.mean(y_pred == y_test)))
print("| Правильность модели через knn.score:  {:.2f}".format(knn.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("| Правильность k-ближайших соседей:     {:.2f}".format(knn.score(X_test, y_test)))