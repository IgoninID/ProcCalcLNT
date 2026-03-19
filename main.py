__author__ = 'Игонин В.Ю.'

from LNT import *


if __name__ == "__main__":
    print("Версия torch:", torch.__version__)
    print("CUDA доступен?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Устройство:", torch.cuda.get_device_name(0))
        print("CUDA версия :", torch.version.cuda)

    # Подберите размеры под свой компьютер
    # Хорошие варианты на 2024–2026 год:
    #   RTX 3060/4060/4070/4080/4090 → 3000–6000
    #   RTX 3060 laptop / 4060 → 2000–4000
    #   MacBook M2/M3 Pro/Max    → 2000–5000
    #   обычный ноутбук i7/Ryzen  → 1000–2000

    benchmark_matmul(
    sizes = [500, 1000, 2000, 3000],
    repeats = 5
    )
