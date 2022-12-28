
def add(a:int, b:int) -> int:

    return a + b


def add_v2(a:int, b:int) -> int:

    if not isinstance(a, int) and not isinstance(b, int):
        raise TypeError("Inputs must be integers")

    return a + b