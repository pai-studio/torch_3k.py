from .backend import cuda_available


def is_available():
    return cuda_available()
