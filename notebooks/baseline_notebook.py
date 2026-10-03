"""ЭТО КОД ИЗ НОУТБУКА. Он работает — и это всё хорошее, что о нём можно сказать.

Задача занятия 1: превратить его в модуль src/train.py, который можно
запустить командой, воспроизвести завтра и передать коллеге.

Найдите здесь как минимум семь проблем, мешающих вывести это в прод.
Список того, что нашли, — часть домашнего задания.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/anna/Desktop/работа/данные/churn_final_v3_ПОСЛЕДНИЙ.csv")

df = df.dropna()

df["contract_type"] = df["contract_type"].map({"month-to-month": 0, "one_year": 1, "two_year": 2})
df["internet_service"] = df["internet_service"].map({"fiber": 0, "dsl": 1, "none": 2})
df["payment_method"] = df["payment_method"].map(
    {"electronic_check": 0, "mailed_check": 1, "bank_transfer": 2, "credit_card": 3}
)

X = df.drop(["churn", "customer_id"], axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=157, max_depth=9, min_samples_leaf=3)
model.fit(X_train, y_train)

print("auc:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# сохранила лучшую, не удалять!!!
# import pickle
# pickle.dump(model, open("model_v2_FINAL.pkl", "wb"))
