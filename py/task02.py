"""
Решение задачи 2.
"""


from os import path
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({
#    "font.family": "Computer Modern Roman"
# })


def convert(array, type1, type2):
    """
    Преобразует массив из 2 элементов к типам (type1, type2) соответственно
    """
    if len(array) != 2:
        raise ValueError(f"Длина array должна быть 2. Передан {array}")
    return [type1(array[0]), type2(array[1])]


def model1(data):
    """
    Подгоняет данные к модели P = a*ln(V)
    """
    v = np.array(list(data.keys()))
    p = np.array(list(data.values()))
    ln_v = np.log(v)

    # значение, при котором минимизируется абс. погрешность
    a = p.dot(ln_v) / ln_v.dot(ln_v)

    # вычисление меры соответствия
    p_adjusted = a * ln_v
    xi = (p - p_adjusted) / np.sqrt(p_adjusted)

    # минимальная абсолютная погрешность
    min_abs_error = (p.dot(p) * ln_v.dot(ln_v) -
                     (p.dot(ln_v))**2) / ln_v.dot(ln_v)
    print(f"Abs.error1 = {min_abs_error}, xi1**2 = {xi.dot(xi)}")
    return xi.dot(xi), a


def model2(data):
    """
    Подгоняет данные к модели P = a*V**b
    """
    x = sp.symbols("x")

    v = np.array(list(data.keys()))
    p = np.array(list(data.values()))
    vx = v ** x

    fun1_ = -(p.dot(vx)) ** 2 / (p.dot(p) * vx.dot(vx))
    fun = sp.lambdify(x, fun1_)

    optimization = minimize_scalar(fun)
    if not optimization.success:
        raise RuntimeError(f"Ошибка оптимизации:\n{optimization.message}")
    b = optimization.x

    vb = v ** b
    a = p.dot(vb) / vb.dot(vb)

    p_adjusted = a * vb

    abs_error_vector = p - p_adjusted
    xi_root = (p - p_adjusted) / np.sqrt(p_adjusted)

    # минимальная абсолютная погрешность
    min_abs_error = abs_error_vector.dot(abs_error_vector)
    print(f"Abs.error2 = {min_abs_error}, xi2**2 = {xi_root.dot(xi_root)}")

    return xi_root.dot(xi_root), a, b


def show_data(data):
    """
    ХУДОжникъ, господа!
    """
    v = np.array(list(data.keys()))
    p = np.array(list(data.values()))

    k = 1.2
    n = 201  # кол-во отрезков разбиения + 1
    v_range = np.linspace(1e-5, max(v) * k, n)
    p_range = np.linspace(0, max(p) * k, n)

    xi1, a1 = model1(data)
    xi2, a2, b2 = model2(data)

    print(
        f"Модель a1*ln(V):\ta1 = {a1}\nМодель a2*v**b2:\ta2 = {a2}, b2 = {b2}")

    accuracy = xi1 / xi2
    if accuracy < 1 and accuracy != 0:
        accuracy = 1 / accuracy
        # accuracy *= 100
    print(f"Точность соответствия моделей равна {accuracy:.4f}")

    curve1 = a1 * np.log(v_range)  # P = a1 * ln(V)
    curve2 = a2 * v_range ** b2  # P = a2 * V ** b

    plt.figure(figsize=(6, 6))
    plt.title("Сравнение моделей")
    plt.grid(visible=1, color="#d4d4d4")

    # Если вознкнет какая-то ошибка, то она связана с отсутствием LaTeX в системе
    # В этом случае либо добавьте путь к latex в переменную PATH,
    # Либо измените строки, переданные в label ниже
    plt.plot(v, p, linestyle="none", color="#dbab39",
             marker="o", label="Исходные данные")
    plt.plot(v_range, curve1, color="#cf4e4e",
             label=rf"Модель 1: $P \approx {a1:.4f} \cdot \ln V$")
    plt.plot(v_range, curve2, color="#3fd4b4",
             label=r"Модель 2: $P \approx %.4f \cdot V^{%.4f}$" % (a2, b2))

    plt.xlim(min(v_range), max(v_range))
    plt.ylim(min(p_range), max(p_range))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    FILENAME = "task2_data"
    SEP = "/"
    data_location = path.dirname(__file__) + SEP + "data"
    DATAFILE_PATH = SEP.join([data_location, FILENAME])

    # чтение файла
    with open(DATAFILE_PATH, "r", encoding="utf-8") as datafile:
        dataset = datafile.read().split("\n")  # запись данных
        dataset = dataset[1:-1]  # заголовки не нужны
        dataset = [convert(elem.split(), float, float)
                   for elem in dataset]  # преобразование к нужным типам
        dataset = dict(dataset)  # {V: P}

    show_data(dataset)
