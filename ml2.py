import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TRAIN_PATH = "train-dset.parquet"
TEST_PATH = "test-dset-small.parquet"
SUBMISSION_PATH = "solution.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Использую:
    - поведенческий сигнал (click conversion)
    - совпадение категорий
    - совпадение локации
    - логарифм цены
    """
    df = df.copy()

    # Работа с пропусками
    df["item_query_click_conv"] = df["item_query_click_conv"].fillna(0)
    df["price"] = df["price"].fillna(0)

    # Бинарные признаки совпадения
    df["same_cat"] = (df.query_cat == df.item_cat_id).astype(int)
    df["same_loc"] = (df.query_loc == df.item_loc).astype(int)

    # Цена в логарифмическом масштабе устойчивее
    df["price_log"] = np.log1p(df["price"])

    # Сильный исторический сигнал
    df["click_score"] = df["item_query_click_conv"]

    return df

print("Загрузка данных")

train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
test = pd.read_parquet(TEST_PATH, engine="pyarrow")

train = build_features(train)
test = build_features(test)


FEATURES = [
    "click_score",
    "same_cat",
    "same_loc",
    "price_log"
]

X_train = train[FEATURES]
y_train = train["item_contact"]

X_test = test[FEATURES]

"""
Использую логистическую регрессию как классическую ML-модель.

 она:
- простая и понятная
- быстро обучается
- устойчива к шуму
"""

model = Pipeline([
    ("scaler", StandardScaler()),# нормализация признаков
    ("lr", LogisticRegression(max_iter=1000))
])

print("Обучение модели...")
model.fit(X_train, y_train)


print("Подсчет релевантности...")

# Вероятность контакта = скор релевантности
test["score"] = model.predict_proba(X_test)[:, 1]

# Сортирую объявления внутри каждого запроса
submission = (
    test
    .sort_values(["query_id", "score"], ascending=[True, False])
    [["query_id", "item_id"]]
)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Файл сохранён: {SUBMISSION_PATH}")
