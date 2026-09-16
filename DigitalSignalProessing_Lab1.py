import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Вхідні параметри
N_STUDENT = 6       # Номер у журналі n = 6 
ORDER_N = 10        # Кількість гармонік N = 10 
A_LIMIT = -np.pi    # Інтервал [-pi, pi]
B_LIMIT = np.pi
W0 = 1.0            # Базова частота w0 = 2*pi / (2*pi) = 1.0



#Точне аналітичне обчислення f(x)
def target_function(x):
    """
    f(x) = 6 * x * exp(-x^2 / 6).
    Функція є непарною на [-pi, pi].
    """
    return N_STUDENT * x * np.exp(- (x ** 2) / N_STUDENT)


def integrand_cos(x, k):
    """Підінтегральна функція для коефіцієнта a_k: f(x) * cos(k*x)"""
    return target_function(x) * np.cos(k * W0 * x)


def integrand_sin(x, k):
    """Підінтегральна функція для коефіцієнта b_k: f(x) * sin(k*x)"""
    return target_function(x) * np.sin(k * W0 * x)


# Обчислення коефіцієнтів a_k та b_k
def compute_fourier_coefficients(N):
    a_coeffs = np.zeros(N + 1)
    b_coeffs = np.zeros(N + 1)

    # a_0 = (1 / pi) * integral(f(x) dx) від -pi до pi
    int_a0, _ = quad(target_function, A_LIMIT, B_LIMIT)
    a_coeffs[0] = (1.0 / np.pi) * int_a0

    # Обчислення a_k та b_k 
    for k in range(1, N + 1):
        int_a, _ = quad(integrand_cos, A_LIMIT, B_LIMIT, args=(k,))
        a_coeffs[k] = (1.0 / np.pi) * int_a

        int_b, _ = quad(integrand_sin, A_LIMIT, B_LIMIT, args=(k,))
        b_coeffs[k] = (1.0 / np.pi) * int_b

    return a_coeffs, b_coeffs


#Наближення рядом Фур'є порядку N
def fourier_approximation(x, a_coeffs, b_coeffs, N):
    s_val = a_coeffs[0] / 2.0
    for k in range(1, N + 1):
        s_val += a_coeffs[k] * np.cos(k * W0 * x) + b_coeffs[k] * np.sin(k * W0 * x)
    return s_val


# ЗАВДАННЯ 5: Оцінка відносної похибки наближення (у нормі L2)
def calculate_relative_error(a_coeffs, b_coeffs, N, num_points=2000):
    x = np.linspace(A_LIMIT, B_LIMIT, num_points)
    y_exact = target_function(x)
    y_approx = fourier_approximation(x, a_coeffs, b_coeffs, N)

    err_norm = np.sqrt(np.sum((y_exact - y_approx) ** 2))
    exact_norm = np.sqrt(np.sum(y_exact ** 2))
    return (err_norm / exact_norm) * 100.0


# ЗАВДАННЯ 6: Збереження у файл
def save_results_to_file(filename, N, a_coeffs, b_coeffs, rel_error):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Функція: f(x) = {N_STUDENT} * x * exp(-x^2 / {N_STUDENT})\n")
        f.write(f"Інтервал: [-pi, pi]\n")
        f.write(f"Порядок наближення N = {N}\n")
        f.write(f"Відносна похибка: {rel_error:.4f} %\n\n")
        f.write("Коефіцієнти Фур'є:\n")
        f.write(f"{'k':>4} | {'a_k (косинус)':>16} | {'b_k (синус)':>16}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{0:>4} | {a_coeffs[0]:>16.6e} | {'0.000000':>16}\n")
        for k in range(1, N + 1):
            f.write(f"{k:>4} | {a_coeffs[k]:>16.6e} | {b_coeffs[k]:>16.6f}\n")
    
    print(f"Дані збережено у файл: '{filename}'")


#Побудова графіків
def plot_results(N, a_coeffs, b_coeffs):
    x = np.linspace(A_LIMIT, B_LIMIT, 1000)
    y_exact = target_function(x)
    y_approx = fourier_approximation(x, a_coeffs, b_coeffs, N)

    fig, axs = plt.subplots(3, 1, figsize=(9, 11))
    fig.suptitle(f"Лабораторна робота №1 (n = {N_STUDENT}, N = {N})", fontsize=13, fontweight='bold')

    #Порівняння сигналів
    axs[0].plot(x, y_exact, label=r"Точна $f(x) = 6x e^{-x^2/6}$", color='black', linewidth=2)
    axs[0].plot(x, y_approx, label=f"Ряд Фур'є (N={N})", color='red', linestyle='--', linewidth=1.8)
    axs[0].set_title("Наближення функції на відрізку $[-\\pi, \\pi]$")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("f(x)")
    axs[0].grid(True, linestyle=':')
    axs[0].legend()

    #Частотний спектр a_k
    k_vals = np.arange(N + 1)
    axs[1].stem(k_vals, a_coeffs, linefmt='b-', markerfmt='bo', basefmt='k-')
    axs[1].set_title(r"Частотний спектр косинусних складових $a(k)$")
    axs[1].set_xlabel("Номер гармоніки k")
    axs[1].set_ylabel(r"$a_k$")
    axs[1].grid(True, linestyle=':')
    axs[1].set_xticks(k_vals)

    #Частотний спектр b_k
    k_sin = np.arange(1, N + 1)
    axs[2].stem(k_sin, b_coeffs[1:], linefmt='g-', markerfmt='go', basefmt='k-')
    axs[2].set_title(r"Частотний спектр синусних складових $b(k)$")
    axs[2].set_xlabel("Номер гармоніки k")
    axs[2].set_ylabel(r"$b_k$")
    axs[2].grid(True, linestyle=':')
    axs[2].set_xticks(k_vals)

    plt.tight_layout()
    plt.savefig("fourier_plots.png", dpi=300)
    print("Графіки збережено як 'fourier_plots.png'")
    plt.show()



#Головна програма
def main():
    
    a_coeffs, b_coeffs = compute_fourier_coefficients(ORDER_N)
    rel_error = calculate_relative_error(a_coeffs, b_coeffs, ORDER_N)
    
    print(f"Відносна похибка апроксимації: {rel_error:.4f}%")
    
    save_results_to_file("fourier_results.txt", ORDER_N, a_coeffs, b_coeffs, rel_error)
    plot_results(ORDER_N, a_coeffs, b_coeffs)


if __name__ == "__main__":
    main()