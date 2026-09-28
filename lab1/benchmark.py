import subprocess
import os
import matplotlib.pyplot as plt

THREADS_LIST = [1, 2, 4, 16, 32, 64]
EXEC_NAME = "./OpenMPlab1"

if not os.path.exists(EXEC_NAME):
    print(f"Ошибка: Сначала скомпилируйте программу! Файл {EXEC_NAME} не найден.")
    exit(1)

times = []
speedups = []
efficiencies = []

print("=" * 60)
print(f" Запуск бенчмарка для {EXEC_NAME} ")
print("=" * 60)

t1 = None
for p in THREADS_LIST:
    print(f"Запуск на {p:2d} потоках... ", end="", flush=True)

    result = subprocess.run([EXEC_NAME, str(p)], capture_output=True, text=True)
    output = result.stdout
    t_p = None

    for line in output.split('\n'):
        if "THREADS:" in line and "TIME:" in line:
            try:
                parts = line.split("TIME:")
                t_p = float(parts[1].strip())
                break
            except (IndexError, ValueError):
                continue

    if t_p is None:
        print(f"\n\nОшибка парсинга! Вывод вашей программы был следующим:\n{output}")
        exit(1)

    times.append(t_p)
    if p == 1:
        t1 = t_p

    sp = t1 / t_p
    ep = (sp / p) * 100

    speedups.append(sp)
    efficiencies.append(ep)
    print(f"Успешно. Время: {t_p:.6f} сек.")

print("\n" + "=" * 65)
print(f" {'Потоки (p)':<12} | {'Время Т(p), сек':<16} | {'Ускорение S(p)':<14} | {'Эффективность E(p)':<12} ")
print("-" * 65)
for i, p in enumerate(THREADS_LIST):
    print(f" {p:<12} | {times[i]:<16.6f} | {speedups[i]:<14.2f} | {efficiencies[i]:.1f}% ")
print("=" * 65)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(THREADS_LIST, THREADS_LIST, 'g--', label="Идеальное ускорение ($S_p = p$)")
ax1.plot(THREADS_LIST, speedups, 'ro-', label="Экспериментальное ускорение", linewidth=2)
ax1.set_title("График ускорения $S_p$")
ax1.set_xlabel("Количество потоков (p)")
ax1.set_ylabel("Ускорение ($T_1 / T_p$)")
ax1.grid(True, linestyle=':')
ax1.legend()

ax2.plot(THREADS_LIST, efficiencies, 'bo-', label="Эффективность", linewidth=2)
ax2.set_title("График эффективности $E_p$")
ax2.set_xlabel("Количество потоков (p)")
ax2.set_ylabel("Эффективность (%)")
ax2.set_ylim(0, 110)
ax2.grid(True, linestyle=':')
ax2.legend()

# Сохранение результатов в графический файл
output_image = "scalability_charts.png"
plt.tight_layout()
plt.savefig(output_image, dpi=150)
print(f"\n Графики успешно сохранены в файл: {output_image}")
