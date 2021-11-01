# Дан признак с номинальными классами, который не имеет внутренней упорядоченности (например, яблоко, груша, банан).

# Преобразовать признак в кодировку с одним активным состоянием1 с помощью
# класса LabeiBinarizer библиотеки scikit-leam:

import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

# Создать кодировщик одного активного состояния
one_hot = LabelBinarizer()

# Преобразовать признак в кодировку с одним активным состоянием
print(one_hot.fit_transform(feature))

# Для вывода классов можно воспользоваться атрибутом classes :
print(one_hot.classes_)

# Если требуется обратить кодирование с одним активным состоянием, то можно
# применить метод inverse_transform:
print(one_hot.inverse_transform(one_hot.transform(feature)))


# Одной из полезных возможностей библиотеки scikit-leam является обработка
# ситуации, когда в каждом наблюдении перечисляется несколько классов:
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# Создать мультиклассовый кодировщик, преобразующий признак в кодировку с одним активным состоянием
one_hot_multiclass = MultiLabelBinarizer()

# Кодировать мультиклассовый признак в кодировку с одним активным состоянием
print(one_hot_multiclass.fit_transform(multiclass_feature))
print(one_hot_multiclass.classes_)