# import sklearn
# import matplotlib
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


# ================  FORGE  ================
# ------ Двухклассовая классификация ------
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
plt.title("FORGE")
print("\n\nФорма массива data для набора Forge: {}\n\n".format(X.shape))
plt.show()


# ================  WAVE  ================
# --------------- Регрессия --------------
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.title("WAVE")
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.show()


# ============  BREAST CANCER  ============
# ------ Real data from scikit-learn ------
cancer = load_breast_cancer()
print("\n\nКлючи cancer(): \n{}".format(cancer.keys()))
print("Форма массива data для набора cancer: {}".format(cancer.data.shape))
print("Количество примеров для каждого класса:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
# print("Имена признаков:\n{}".format(cancer.feature_names))


# ===========  BOSTON HOUSING  ===========
# ---- Реальные данные для регрессии -----
boston = load_boston()
print("\n\nФорма массива data для набора boston: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()              # dataset с производными признаками [feature engineering]
print("Форма массива data для boston_extend: {}\n\n".format(X.shape))


# =========  FORGE K-NEIGHBOURS =========
# ----- Алгоритм K-ближайших соседей ----
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.title("FORGE 1-NEIGHBOUR")
plt.show()
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.title("FORGE 3-NEIGHBOUR")          # для присвоения метки используется голосование (voting)
plt.show()


# ========  SCIKIT K-NEIGHBOURS =========
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("\n\nScikit-learn K-neighbours:")
print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))
print("Правильность на тестовом наборе: {:.2f}\n\n".format(clf.score(X_test, y_test)))


# ========  DECISION BOUNDARY ==========
#   Граница принятия решений для forge
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("Количество соседей: {}".format(n_neighbors))
    ax.set_xlabel("признак 0")
    ax.set_ylabel("признак 1")
axes[0].legend(loc=3)
plt.show()


# ========  СЛОЖНОСТЬ МОДЕЛИ ==========
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="Правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="Правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("Количество соседей")
plt. legend()
plt.show()


# =======  РЕГРЕССИЯ K-NEIGHBOURS =======
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# KNeighborsRegressor — аналогичная функция для алгоритма k-соседей при регрессии
reg = KNeighborsRegressor(n_neighbors=3)
reg. fit(X_train, y_train)
print("\n\nПрогнозы для тестового набора:\n{}". format(reg.predict(X_test)))
# R^2 — коэффициент детерминации — качество регрессионной модели
print("R^2 на тестовом наборе: {:.2f}\n\n".format(reg.score(X_test, y_test)))


# ==========  АНАЛИЗ РЕГРЕССИИ ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
        n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("Признак")
    ax.set_ylabel("Целевая переменная")
axes[0].legend(["Прогнозы модели", "Обучающие данные/ответы", "Тестовые данные/ответы"], loc="best")
plt.show()