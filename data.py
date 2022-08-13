import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
data = pd.read_csv('train (2).csv')
print(data.head(4))

# Бинарная классификация, предсказываем 2 класса
dict_live = {
    0: 'Perished',
    1: 'Survived'
}

# Параметр пола в словарь
dict_sex = {
    'male': 0,
    'female': 1
}

data['Bsex'] = data['Sex'].apply(lambda x: dict_sex[x])

# Формируем признаи и целевые переменные
features = data[['Pclass', 'Bsex']].to_numpy()
labels = data['Survived'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)

# print('Training records:', Y_train.size)
# print('Test records:', Y_test.size)
