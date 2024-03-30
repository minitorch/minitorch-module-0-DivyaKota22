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
    return 1.0 / (1.0 + math.exp(-x))

def relu(x: float) -> float:
    """ReLU function."""
    return x if x > 0 else 0.0

# Remaining functions (log_back, inv, inv_back, relu_back) assume knowledge of derivatives.
#0.3
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a function to each element in the iterable."""
    def apply_map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return apply_map

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in the list."""
    return map(neg)(ls)

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies a function to pairs of elements from two lists."""
    def apply_zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply_zipWith

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists."""
    return zipWith(add)(ls1, ls2)

def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """Reduce the list to a single value by recursively applying a function."""
    def apply_reduce(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return apply_reduce

def sum(ls: Iterable[float]) -> float:
    """Sum all elements of the list."""
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    """Multiply all elements of the list."""
    return reduce(mul, 1.0)(ls)
