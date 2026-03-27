__author__ = 'Игонин В.Ю.'

from LNT import *


if __name__ == "__main__":
    print("Версия torch:", torch.__version__)
    print("CUDA доступен?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Устройство:", torch.cuda.get_device_name(0))
        print("CUDA версия :", torch.version.cuda)

    bench_matmul(
    sizes = [500, 1000, 1200, 2000, 3000, 5000],
    repeats = 5
    )
