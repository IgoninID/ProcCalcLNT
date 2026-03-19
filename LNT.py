__author__ = 'Игонин В.Ю.'

import time
import numpy as np
import torch


def multiply_lists(A, B):
    """Чистый Python, списки списков"""
    n = len(A)
    m = len(B[0])
    p = len(B)

    # C = A × B
    C = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C


def benchmark_matmul(sizes=[800, 1200, 2000, 3000], repeats=3):
    print("\nСравнение времени матричного умножения (A @ B)")
    print("Повторений для каждого размера:", repeats, "\n")

    header = f"{'Размер':>8}  {'Python lists':>14}   {'numpy':>12}   {'torch CPU':>12}   {'torch CUDA':>12}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        # Подготовка данных
        A_np = np.random.randn(n, n).astype(np.float32)
        B_np = np.random.randn(n, n).astype(np.float32)

        A_torch_cpu = torch.from_numpy(A_np)
        B_torch_cpu = torch.from_numpy(B_np)

        A_torch_gpu = None
        B_torch_gpu = None
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            A_torch_gpu = A_torch_cpu.cuda()
            B_torch_gpu = B_torch_cpu.cuda()

        # Списки
        A_list = A_np.tolist()
        B_list = B_np.tolist()

        # Замеры
        times = {"lists": [], "numpy": [], "torch_cpu": [], "torch_gpu": []}

        # Python lists
        if n <= 1200:
            t0 = time.perf_counter()
            multiply_lists(A_list, B_list)
            t1 = time.perf_counter()
            times["lists"].append(t1 - t0)

        # numpy
        for _ in range(repeats):
            torch.cuda.synchronize() if has_cuda else None
            t0 = time.perf_counter()
            C = A_np @ B_np
            t1 = time.perf_counter()
            times["numpy"].append(t1 - t0)

        # torch CPU
        for _ in range(repeats):
            t0 = time.perf_counter()
            C = A_torch_cpu @ B_torch_cpu
            t1 = time.perf_counter()
            times["torch_cpu"].append(t1 - t0)

        # torch GPU
        if has_cuda:
            torch.cuda.synchronize()
            for _ in range(repeats):
                t0 = time.perf_counter()
                C = A_torch_gpu @ B_torch_gpu
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times["torch_gpu"].append(t1 - t0)

        # Вывод результатов
        row = f"{n:8d}  "

        if times["lists"]:
            t = np.mean(times["lists"])
            row += f"{t:12.3f} s   "
        else:
            row += f"{'--- слишком медленно ---':>14}   "

        row += f"{np.mean(times['numpy']):12.3f} s   "
        row += f"{np.mean(times['torch_cpu']):12.3f} s   "

        if has_cuda and times["torch_gpu"]:
            row += f"{np.mean(times['torch_gpu']):12.3f} s"
        elif has_cuda:
            row += "   --- error ---"
        else:
            row += "      нет GPU"

        print(row)