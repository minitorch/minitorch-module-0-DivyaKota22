import math
from typing import Callable, Iterable

# Implementation of elementary functions for deep learning tasks.

def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

def id(x: float) -> float:
    """Identity function."""
    return x

def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

def neg(x: float) -> float:
    """Negate the number."""
    return -x

def lt(x: float, y: float) -> float:
    """Check if the first number is less than the second."""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """Check if two numbers are equal."""
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    """Check if two numbers are approximately equal."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def relu(x: float) -> float:
    """ReLU function."""
    return max(x, 0.0)

def log_back(x: float, d: float) -> float:
    """Backward pass for the log function."""
    return d / (x + EPS)

def inv(x: float) -> float:
    """Inverse function."""
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    """Backward pass for the inverse function."""
    return -d / (x**2)

def relu_back(x: float, d: float) -> float:
    """Backward pass for the ReLU function."""
    return d if x > 0 else 0.0

# Task 0.3 implementations

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function."""
    def mapped(ls):
        return [fn(x) for x in ls]
    return mapped

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate a list of numbers."""
    return map(neg)(ls)

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (map2) function."""
    def zipped(ls1, ls2):
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return zipped

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise."""
    return zipWith(add)(ls1, ls2)

def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function."""
    def reduced(ls):
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return reduced

def sum(ls: Iterable[float]) -> float:
    """Sum up a list."""
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    """Product of a list."""
    return reduce(mul, 1.0)(ls)
