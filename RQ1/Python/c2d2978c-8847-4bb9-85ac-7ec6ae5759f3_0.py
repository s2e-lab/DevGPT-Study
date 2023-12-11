import math
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

# Define the type variables
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

class Option(Generic[A], ABC):
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def get(self) -> A:
        pass

class Empty(Option[A]):
    def is_empty(self) -> bool:
        return True

    def get(self) -> A:
        raise Exception("Cannot get value from an Empty")

class Value(Option[A]):
    def __init__(self, value: A) -> None:
        self.value = value

    def is_empty(self) -> bool:
        return False

    def get(self) -> A:
        return self.value

# A Partial function is a function that may not provide an answer
PartialFunction = Callable[[A], Option[B]]

# Identity morphism
def identity(a: A) -> Option[A]:
    return Value(a)

# Composition of morphisms
def compose(m1: PartialFunction[A, B], m2: PartialFunction[B, C]) -> PartialFunction[A, C]:
    def composed(a: A) -> Option[C]:
        tmp = m1(a)
        if tmp.is_empty():
            return Empty()
        else:
            return m2(tmp.get())
    return composed

# Safe square root function
def safe_square_root(x: float) -> Option[float]:
    return Value(math.sqrt(x)) if x >= 0 else Empty()

# Safe reciprocal function
def safe_reciprocal(x: float) -> Option[float]:
    return Value(1 / x) if x != 0 else Empty()

# Compose safe_square_root and safe_reciprocal
# This function will first take the reciprocal of a number, and then take the square root of the result
composed_function = compose(safe_reciprocal, safe_square_root)

# Now we can test our composed function

def test():
    # Test with some values
    print(composed_function(4).get())  # Should print 0.5
    print(composed_function(-4).is_empty())  # Should print True because sqrt is not defined for negative numbers
    print(composed_function(0).is_empty())  # Should print True because reciprocal is not defined for 0

test()
