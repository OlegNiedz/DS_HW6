import numpy as np


def hypothesis(X, w):
    """
    Гіпотеза лінійної регресії у векторному вигляді.

    Параметри:
    X : numpy array
        Матриця ознак (рядки - приклади, стовпці - ознаки).
    w : numpy array
        Вектор параметрів моделі.

    Повертає:
    numpy array
        Прогнозовані значення.
    """
    return np.dot(X, w)


def loss_function(y_true, y_pred):
    """
    Функція втрат у векторному вигляді (середньоквадратична помилка).

    Параметри:
    y_true : numpy array
        Реальні значення.
    y_pred : numpy array
        Прогнозовані значення.

    Повертає:
    float
        Значення функції втрат.
    """
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent_step(X, y, w, learning_rate):
    """
    Один крок градієнтного спуску.

    Параметри:
    X : numpy array
        Матриця ознак (рядки - приклади, стовпці - ознаки).
    y : numpy array
        Вектор реальних значень.
    w : numpy array
        Вектор параметрів моделі.
    learning_rate : float
        Швидкість навчання.

    Повертає:
    numpy array
        Оновлені параметри моделі.
    """
    predictions = hypothesis(X, w)
    gradient = np.dot(X.T, predictions - y) / len(y)
    w -= learning_rate * gradient
    return w


# Згенеруємо датасет
np.random.seed(0)
m = 100
X = 2 * np.random.rand(m, 3)
y = 4 + np.dot(X, np.array([3, 5, 2])) + np.random.randn(m)

# Ініціалізуємо параметри моделі
w = np.random.randn(3)

# Гіперпараметри
learning_rate = 0.01
n_iterations = 1000

# Градієнтний спуск
for iteration in range(n_iterations):
    w = gradient_descent_step(X, y, w, learning_rate)

print("Параметри w, знайдені за допомогою градієнтного спуску:", w)

# Аналітичне рішення
X_b = np.c_[np.ones((m, 1)), X]
w_analytical = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Параметри w, знайдені за допомогою аналітичного рішення:", w_analytical)

# Перевірка з scikit-learn
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(
    "Параметри w, знайдені за допомогою scikit-learn:",
    np.append(lin_reg.intercept_, lin_reg.coef_[1:]),
)
