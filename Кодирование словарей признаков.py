# Дан словарь, и требуется его конвертировать в матрицу признаков.

from sklearn.feature_extraction import DictVectorizer

data_dict = [{"красный": 2, "синий": 4},
             {"красный": 4, "синий": 3},
             {"красный": 1, "желтый": 2},
             {"красный": 2, "желтый": 2}]

# Создать векторизатор словаря
dictvectorizer = DictVectorizer(sparse=False)

# Конвертировать словарь в матрицу признаков
features = dictvectorizer.fit_transform(data_dict)
print(features)


# По умолчанию Dictvectorizer выводит разреженную матрицу, в которой хранятся только элементы со значением,
# отличным от 0. Это может быть очень полезно, когда имеются массивные матрицы (часто встречающиеся в обработке
# естественного языка) и требуется минимизировать потребности в оперативной памяти. Мы можем заставить Dictvectorizer
# вывести плотную матрицу, используя sparse=False.

# Имена каждого созданного признака можно получить с помощью метода
features_name = dictvectorizer.get_feature_names()
print(features_name)

# Хотя это и не обязательно, для иллюстрации мы можем создать фрейм данных
# pandas, чтобы результат лучше выглядел
import pandas as pd

print(pd.DataFrame(features, columns=features_name))