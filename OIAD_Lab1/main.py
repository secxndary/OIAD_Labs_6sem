import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
from sklearn.tree import plot_tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay




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
print("===\tОтветы (target):\n{}".format(iris_dataset['target'][:150]))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("\n")
print("======\tФорма массива X_train: {}".format(X_train.shape))
print("======\tФорма массива y_train: {}".format(y_train.shape))
print("======\tФорма массива X_test:  {}".format(X_test.shape))
print("======\tФорма массива y_test:  {}".format(y_test.shape))

# Создаем dataframe из данных в массиве X_train
# Маркируем столбцы, используя строки в iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()





# ниже расположен код с сайта sckit-learn
# он рисует графики дерева решений и поверхности решений

iris = load_iris()
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")


plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()