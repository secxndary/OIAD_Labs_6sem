import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
y_named = np.array(["blue", "red"])[y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
    train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# Бинарная классификация
print("\n=====  DECISION FUNCTION  =====")
print("Форма массива X_test: {}".format(X_test.shape))
print("Форма решающей функции: {}".format(gbrt.decision_function(X_test).shape))

print("\nРешающая функция:\n{}".format(gbrt.decision_function(X_test)[:6]))

print("\nРешающая функция с порогом отсечения:\n{}".format(gbrt.decision_function(X_test) > 0))
print("\nПрогнозы:\n{}".format(gbrt.predict(X_test)))

greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
pred = gbrt.classes_[greater_zero]
print("\npred идентичен прогнозам?\t{}".format(np.all(pred == gbrt.predict(X_test))))

decision_function = gbrt.decision_function(X_test)
print("Решающая функция минимум: {:.2f} максимум: {:.2f}\n".format(
    np.min(decision_function), np.max(decision_function)))


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)

for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel("Характеристика 0")
    ax.set_ylabel("Характеристика 1")
    cbar = plt.colorbar(scores_image, ax=axes.tolist())
    axes[0].legend(["Тест класс 0", "Тест класс 1", "Обучение класс 0", "Обучение класс 1"], ncol=4, loc=(.1, 1.1))
plt.show()




print("\n======  PREDICT PROBA  ======")
print("Форма вероятностей: {}".format(gbrt.predict_proba(X_test).shape))
print("Спрогнозированные вероятности: \n{}\n".format(gbrt.predict_proba(X_test[:6])))


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(
    gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')


for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_ylabel("Характеритика 0")
    ax.set_ylabel("Характеритика 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["Тест класс 0", "Тест класс 1", "Обуч класс 0", "Обуч класс 1"], ncol=4, loc=(.1, 1.1))
plt.show()




# Мультиклассивая классификация
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("\n========  DECISION FUNCTION  ========")
print("Форма решающей функции: {}".format(gbrt.decision_function(X_test).shape))
print("Решающая функция: \n{}".format(gbrt.decision_function(X_test)[:6, :]))

print("\nArgmax решающей функции: \n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
print("Прогнозы: \n{}".format(gbrt.predict(X_test)))


print("\n==========  PREDICT PROBA  ==========")
print("Спрогнозированные вероятности: \n{}".format(gbrt.predict_proba(X_test)[:6]))
print("Суммы: {}". format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))

print("\nArgmax спрогнозированных вероятностей: \n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("Прогнозы: \n{}\n". format(gbrt.predict(X_test)))



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print("\n===============  LOGISTIC REGRESSION  ================")
print("Уникальные классы в обучающем наборе: {}".format(logreg.classes_))
print("Прогнозы: {}".format(logreg.predict(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print("Argmax решающей функции: {}".format(argmax_dec_func[:10]))
print("Argmax объединенный с классами: {}".format(logreg.classes_[argmax_dec_func][:10]))
