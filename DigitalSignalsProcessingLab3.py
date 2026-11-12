import time
import cmath
import random


def compute_dft(x):
    """Класичне ДПФ із підрахунком операцій."""
    n = len(x)
    additions = 0
    multiplications = 0
    result = [0.0j] * n

    for k in range(n):
        for m in range(n):
            angle = -2j * cmath.pi * k * m / n
            result[k] += x[m] * cmath.exp(angle)

            multiplications += 2
            if m > 0:
                additions += 2

    return result, additions, multiplications


def compute_fft(x):
    """Швидке перетворення Фур'є з підрахунком операцій."""
    n = len(x)
    if n <= 1:
        return x, 0, 0

    even, add_e, mult_e = compute_fft(x[0::2])
    odd, add_o, mult_o = compute_fft(x[1::2])

    t = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]

    additions = add_e + add_o + n
    multiplications = mult_e + mult_o + n // 2

    result = [0.0j] * n
    for k in range(n // 2):
        result[k] = even[k] + t[k]
        result[k + n // 2] = even[k] - t[k]

    return result, additions, multiplications


def analyze_algorithms():
    signal_n8 = [0, 1, 1, 0, 0, 1, 1, 0]

    random.seed(42)
    signal_n64 = [random.random() for _ in range(64)]

    datasets = {"N=8": signal_n8, "N=": signal_n64}

    for label, data in datasets.items():
        print(f"\nАналіз для масиву {label}:")

        start_dft = time.perf_counter()
        res_dft, add_dft, mult_dft = compute_dft(data)
        time_dft = time.perf_counter() - start_dft

        start_fft = time.perf_counter()
        res_fft, add_fft, mult_fft = compute_fft(data)
        time_fft = time.perf_counter() - start_fft

        print("k | ДПФ | ШПФ")
        for k in range(len(data)):
            dft_str = f"{res_dft[k].real:8.4f} {res_dft[k].imag:+8.4f}j"
            fft_str = f"{res_fft[k].real:8.4f} {res_fft[k].imag:+8.4f}j"
            print(f"{k:>2} | {dft_str} | {fft_str}")

        print(f"ДПФ -> Час: {time_dft:.6f} сек | Додавання: {add_dft} | Множення: {mult_dft}")
        print(f"ШПФ -> Час: {time_fft:.6f} сек | Додавання: {add_fft} | Множення: {mult_fft}")


if __name__ == "__main__":
    analyze_algorithms()