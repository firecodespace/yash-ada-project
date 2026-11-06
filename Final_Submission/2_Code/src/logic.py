"""Exact syllogism validity by model enumeration (the classical algorithm).

A categorical syllogism talks about three terms S (subject), M (middle) and P (predicate).
Each of the 2^3 = 8 Venn regions (in/out of S, M, P) is either empty or non-empty, so there are
2^8 = 256 possible "worlds". An argument is valid if every world that makes both premises true
also makes the conclusion true: O(256) work per argument.
"""

from functools import lru_cache

S, M, P = 0, 1, 2
# Premise term pairs (major premise first, minor second) for each figure.
FIGURES = {1: [(M, P), (S, M)], 2: [(P, M), (S, M)], 3: [(M, P), (M, S)], 4: [(P, M), (M, S)]}
WORLDS = [frozenset(r for r in range(8) if mask >> r & 1) for mask in range(256)]


def _holds(kind: str, x: int, y: int, world: frozenset) -> bool:
    both = any(r >> x & 1 and r >> y & 1 for r in world)
    only_x = any(r >> x & 1 and not r >> y & 1 for r in world)
    return {"A": not only_x, "E": not both, "I": both, "O": only_x}[kind]


@lru_cache(maxsize=None)
def is_valid(figure: int, mood: str, existential: bool = False) -> bool:
    """Validity of a mood (e.g. 'AAA') in a figure (1-4).

    existential=False: modern (Boolean) reading, terms may be empty.
    existential=True: traditional reading, every term must have at least one member.
    """
    (a, b), (c, d) = FIGURES[figure]
    for world in WORLDS:
        if existential and not all(any(r >> t & 1 for r in world) for t in (S, M, P)):
            continue
        if _holds(mood[0], a, b, world) and _holds(mood[1], c, d, world) and not _holds(mood[2], S, P, world):
            return False
    return True


def verdict(figure: int, mood: str) -> str:
    """'valid', 'invalid' or 'conditional' (valid only if every term is non-empty)."""
    if is_valid(figure, mood):
        return "valid"
    return "conditional" if is_valid(figure, mood, True) else "invalid"
