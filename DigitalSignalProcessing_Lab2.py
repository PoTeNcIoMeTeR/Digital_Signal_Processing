import numpy as np
import matplotlib.pyplot as plt
import time

STUDENT_N = 6                # Номер варіанта
N_PART1 = 10 + STUDENT_N     # Частина I: N = 16
N_NUM_PART2 = 96 + STUDENT_N # Частина II: 96 + 6 = 102

#ДПФ та ОДПФ
def dft_single_harmonic(signal, k, N):
    
    a_k = 0.0
    b_k = 0.0
    mul_count = 0
    add_count = 0

    for i in range(N):
        angle = 2.0 * np.pi * k * i / N
        a_k += signal[i] * np.cos(angle)
        b_k += signal[i] * np.sin(angle)
        mul_count += 4
        add_count += 2

    a_k = a_k / N
    b_k = -b_k / N
    mul_count += 2

    return a_k, b_k, mul_count, add_count

#ДПФ
def compute_dft(signal):
    """
    Повне пряме дискретне перетворення Фур'є (ДПФ).
    Повертає: комплексний вектор C, дійсні A, B, кількість операцій та час.
    """
    N = len(signal)
    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N, dtype=complex)

    total_muls = 0
    total_adds = 0

    start_time = time.perf_counter()
    for k in range(N):
        a_k, b_k, m_cnt, a_cnt = dft_single_harmonic(signal, k, N)
        A[k] = a_k
        B[k] = b_k
        C[k] = complex(a_k, b_k)
        total_muls += m_cnt
        total_adds += a_cnt
    elapsed_time = time.perf_counter() - start_time

    return C, A, B, total_muls, total_adds, elapsed_time

#ОДПФ
def compute_idft(C_coeffs):
   
    N = len(C_coeffs)
    reconstructed = np.zeros(N)

    for n in range(N):
        val = 0.0 + 0.0j
        for k in range(N):
            angle = 2.0 * np.pi * k * n / N
            val += C_coeffs[k] * np.exp(1j * angle)
        reconstructed[n] = np.real(val)

    return reconstructed



#генерація відліків сигналу
def generate_binary_samples(number):
    bin_str = format(number, '08b')
    samples = np.array([float(bit) for bit in bin_str])
    return samples, bin_str

#відтворення сигналу
def reconstruct_analog_signal(C, t_array, Tc=1.0):
    
    half_N = len(C) // 2
    s_t = np.abs(C[0]) * np.ones_like(t_array)

    # гармоніки від 1 до N/2 - 1
    for k in range(1, half_N):
        mod_k = np.abs(C[k])
        phi_k = np.angle(C[k])
        s_t += 2.0 * mod_k * np.cos(2.0 * np.pi * k * t_array / Tc + phi_k)

    # гранична гармоніка k = N/2
    mod_half = np.abs(C[half_N])
    phi_half = np.angle(C[half_N])
    s_t += mod_half * np.cos(np.pi * len(C) * t_array / Tc + phi_half)

    return s_t


# графіки амплітуди і фаз
def plot_dft_spectra(C, filename="lab2_part1_plots.png"):
    N = len(C)
    k_range = np.arange(N)
    magnitudes = np.abs(C)
    phases = np.angle(C)

    fig, axs = plt.subplots(2, 1, figsize=(9, 8))
    fig.suptitle(f"Частина I: Спектри ДПФ тестового сигналу (N = {N})", fontsize=12, fontweight='bold')

    # спектр амплітуд
    axs[0].stem(k_range, magnitudes, linefmt='b-', markerfmt='bo', basefmt='k-')
    axs[0].set_title(r"Спектр амплітуд $|C_k|$")
    axs[0].set_xlabel("Номер гармоніки k"); axs[0].set_ylabel(r"$|C_k|$")
    axs[0].grid(True, linestyle=':'); axs[0].set_xticks(k_range)

    # спектр фаз
    axs[1].stem(k_range, phases, linefmt='r-', markerfmt='ro', basefmt='k-')
    axs[1].set_title(r"Спектр фаз $\arg C_k$ (радіани)")
    axs[1].set_xlabel("Номер гармоніки k"); axs[1].set_ylabel(r"$\varphi_k$ (рад)")
    axs[1].grid(True, linestyle=':'); axs[1].set_xticks(k_range)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Графіки Частини I збережено в '{filename}'")

# графік відновленого аналогового сигналу
def plot_reconstructed_signal(t_cont, s_analog, t_disc, s_discrete, bin_code, filename="lab2_part2_analog.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(t_cont, s_analog, 'b-', label="Відтворений аналоговий сигнал $s(t)$", linewidth=2)
    plt.stem(t_disc, s_discrete, linefmt='r--', markerfmt='ro', basefmt='k-', label=r"Початкові дискретні відліки $s(nT_\delta)$")
    plt.title(f"Частина II: Відтворений аналоговий сигнал за 8 відліками (Код: {bin_code})", fontweight='bold')
    plt.xlabel(r"Нормований час $t / T_c$")
    plt.ylabel("Амплітуда s(t)")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[+] Графік Частини II збережено в '{filename}'")




def save_report(filename, part1_data, part2_data, part3_data):
    with open(filename, 'w', encoding='utf-8') as f:
        # частина I
        f.write(f"ЧАСТИНА I (N = {part1_data['N']})\n")
        f.write(f"Вхідний сигнал: {part1_data['signal']}\n")
        f.write(f"Час: {part1_data['time']:.2f} мкс | Множень: {part1_data['muls']} | Додавань: {part1_data['adds']}\n")
        f.write(f"{'k':>3} | {'A_k':>10} | {'B_k':>10} | {'|C_k|':>10} | {'arg C_k (рад)':>14}\n" + "-" * 56 + "\n")
        for k in range(part1_data['N']):
            f.write(f"{k:>3} | {part1_data['A'][k]:>10.4f} | {part1_data['B'][k]:>10.4f} | "
                    f"{part1_data['mag'][k]:>10.4f} | {part1_data['phase'][k]:>14.4f}\n")

        # частина II
        f.write(f"\nЧАСТИНА II (N_bin = {part2_data['number']}, двійковий код: {part2_data['code']})\n")
        f.write(f"Відліки: {part2_data['samples']}\n")
        f.write(f"{'k':>3} | {'Re(C_k)':>10} | {'Im(C_k)':>10} | {'|C_k|':>10} | {'arg C_k (рад)':>14}\n" + "-" * 56 + "\n")
        for k in range(5):
            f.write(f"{k:>3} | {part2_data['C'][k].real:>10.4f} | {part2_data['C'][k].imag:>10.4f} | "
                    f"{part2_data['mag'][k]:>10.4f} | {part2_data['phase'][k]:>14.4f}\n")

        # частина III
        f.write(f"\nЧАСТИНА III: ОДПФ\n")
        f.write(f"Початкові відліки:   {part3_data['orig']}\n")
        f.write(f"Відновлені відліки: {part3_data['rec']}\n")
        f.write(f"s(0) розрахункове = {part3_data['s0']:.4f}\n")
        f.write(f"s(1) розрахункове = {part3_data['s1']:.4f}\n")

    print(f"[+] Результати успішно збережено у файл '{filename}'")


def main():
    #частина I (ДПФ тестового сигналу N = 16)
    print(f">>> Виконання Частини I (N = {N_PART1})...")
    np.random.seed(42)
    test_sig = np.round(np.random.uniform(0.0, 5.0, N_PART1), 2)
    C1, A1, B1, muls, adds, t_sec = compute_dft(test_sig)
    plot_dft_spectra(C1)

    print(f"Показники ефективності:")
    print(f"Час: {t_sec * 1e6:.2f} мкс | Множень: {muls} | Додавань: {adds}")

    #частина II (Відтворення сигналу 8 відліків)
    print(f"\n>>> Виконання Частини II (Число {N_NUM_PART2})...")
    s_samples, bin_code = generate_binary_samples(N_NUM_PART2)
    C2, _, _, _, _, _ = compute_dft(s_samples)

    Tc = 1.0
    t_cont = np.linspace(0.0, Tc, 1000)
    s_analog = reconstruct_analog_signal(C2, t_cont, Tc)
    t_disc = np.linspace(0.0, Tc - Tc/8.0, 8)
    plot_reconstructed_signal(t_cont, s_analog, t_disc, s_samples, bin_code)

    print("Комплексні коефіцієнти C_k (k = 0..4):")
    for k in range(5):
        print(f"C_{k} = {C2[k].real:+.4f} {C2[k].imag:+.4f}j | |C_{k}| = {np.abs(C2[k]):.4f} | arg = {np.angle(C2[k]):.4f} рад")

    # частина III (ОДПФ)
    print(f"\n>>> Виконання Частини III (ОДПФ)...")
    s_restored = compute_idft(C2)
    s0_val = np.sum(C2).real
    s1_val = np.sum([C2[k] * np.exp(1j * 2.0 * np.pi * k / 8.0) for k in range(8)]).real

    print(f"Оригінальні відліки:   {s_samples.astype(int).tolist()}")
    print(f"Відновлені через ОДПФ: {np.round(s_restored).astype(int).tolist()}")
    print(f"s(0) = {s0_val:.4f} (точне: {int(s_samples[0])})")
    print(f"s(1) = {s1_val:.4f} (точне: {int(s_samples[1])})")

    #експорт даних
    part1_res = {'N': N_PART1, 'signal': test_sig.tolist(), 'muls': muls, 'adds': adds, 
                 'time': t_sec * 1e6, 'A': A1, 'B': B1, 'mag': np.abs(C1), 'phase': np.angle(C1)}
    part2_res = {'number': N_NUM_PART2, 'code': bin_code, 'samples': s_samples.tolist(), 
                 'C': C2, 'mag': np.abs(C2), 'phase': np.angle(C2)}
    part3_res = {'orig': s_samples.tolist(), 'rec': np.round(s_restored, 4).tolist(), 
                 's0': s0_val, 's1': s1_val}

    save_report("lab2_results.txt", part1_res, part2_res, part3_res)
    
    plt.show()


if __name__ == "__main__":
    main()