# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Частина 1: Завантаження та дослідницький аналіз даних
data = fetch_california_housing(as_frame=True)
df = data.frame
df['Price'] = data.target  # Додаємо колонку з цінами

# Описова статистика
print(df.describe())

# Перевірка на пропущені значення
print(df.isnull().sum())

# Типи даних
print(df.dtypes)

# Візуалізація: гістограми
df.hist(bins=20, figsize=(14, 10))
plt.suptitle("Гістограми розподілу ознак")
plt.show()

# Візуалізація: boxplot для виявлення викидів
plt.figure(figsize=(14, 8))
sns.boxplot(data=df.drop('Price', axis=1))
plt.title("Boxplot для виявлення викидів")
plt.xticks(rotation=45)
plt.show()

# Кореляційна матриця
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Кореляційна матриця")
plt.show()

# Scatter plots між ціною та іншими ознаками
features = df.columns[:-1]
plt.figure(figsize=(16, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    plt.scatter(df[feature], df['Price'], alpha=0.5)
    plt.title(f'{feature} vs Price')
    plt.xlabel(feature)
    plt.ylabel('Price')
plt.tight_layout()
plt.show()

# Частина 2: Підготовка даних
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Частина 3: Побудова моделей
most_correlated_feature = corr_matrix['Price'][:-1].idxmax()
X_train_simple = X_train_scaled[:, X.columns.get_loc(most_correlated_feature)].reshape(-1, 1)
X_test_simple = X_test_scaled[:, X.columns.get_loc(most_correlated_feature)].reshape(-1, 1)

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)

# Результати простої моделі
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
print(f"Проста модель - MSE: {mse_simple}, R²: {r2_simple}")

# Множинна лінійна регресія
model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train)
y_pred_multiple = model_multiple.predict(X_test_scaled)

# Результати множинної моделі
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)
print(f"Множинна модель - MSE: {mse_multiple}, R²: {r2_multiple}")

# Частина 4: Оцінка моделей
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_multiple, alpha=0.5)
plt.plot([0, 5], [0, 5], '--', color='red')
plt.title("Передбачені vs Реальні значення")
plt.xlabel("Реальні ціни")
plt.ylabel("Передбачені ціни")
plt.show()

# Графік залишків
residuals = y_test - y_pred_multiple
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("Розподіл залишків")
plt.show()

# Частина 5: Інтерпретація результатів
def predict_price(features):
    scaled_features = scaler.transform([features])
    return model_multiple.predict(scaled_features)[0]

# Наприклад, прогноз для певних характеристик
sample_features = [5, 30, 6, 1, 3000, 3, 34.0, -118.0]
predicted_price = predict_price(sample_features)
print(f"Прогнозована ціна: {predicted_price}")
