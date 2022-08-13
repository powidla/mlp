import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
import class_mlp
from data import X_train, Y_train, X_test, Y_test, dict_live
warnings.filterwarnings('ignore')
model = class_mlp.MLP()

model.add(class_mlp.layers.layer(24, 'ReLU'))
model.add(class_mlp.layers.layer(12, 'sigmoid'))
model.add(class_mlp.layers.layer(6, 'ReLU'))
model.add(class_mlp.layers.layer(1, 'sigmoid'))

model.set_learning_rate(0.8)

model.Fit(X_train, Y_train)
acc_val = model.get_accuracy()
acc_avg_val = model.get_avg_accuracy()

plt.figure(figsize=(10, 6))
plt.scatter(np.arange(1, len(acc_avg_val) + 1), acc_avg_val, label='mu')
plt.title('Среднее значение функции потерь', fontsize=20)
plt.xlabel('Тренировочная выборка', fontsize=16)
plt.ylabel('Лосс', fontsize=16)
plt.show()

predictions = model.predict(X_test)

# Строим матрицу корреляции предсказаний
cm = confusion_matrix(Y_test, predictions)

df_cm = pd.DataFrame(cm, index=[dict_live[i] for i in range(0, 2)], columns=[dict_live[i] for i in range(0, 2)])
plt.figure(figsize=(7, 7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Предсказания модели по классам", fontsize=18)
plt.ylabel("Правильные классы", fontsize=18)
plt.show()
