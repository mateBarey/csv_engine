from operator import eq, ne, lt, le, gt, ge, add, sub, mul, truediv

CMP_OPS = {
    "eq": eq, "==": eq, "ne": ne, "!=": ne,
    "lt": lt, "<": lt, "le": le, "<=": le,
    "gt": gt, ">": gt, "ge": ge, ">=": ge,
    "in": lambda a, b: b in a,
    "not in": lambda a, b: b not in a
}

LOGICAL_OPS = {
    "and": lambda p1, p2: lambda row: p1(row) and p2(row),
    "or": lambda p1, p2: lambda row: p1(row) or p2(row),
    "not": lambda p: lambda row: not p(row),
}

ARITH_OPS = {
    "+": add, "-": sub, "*": mul, "/": truediv,
    "concat": lambda a, b: str(a) + str(b)
}
