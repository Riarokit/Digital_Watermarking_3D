"""Compiled bipartite matching kernels for geometry-only correspondence.

HK checks cardinality only. Auction minimizes costs approximately on the
already feasible candidate graph; it never uses received vertex identities.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _hk_dfs(row, indptr, indices, row_mate, column_mate, distance, shortest):
    for edge in range(indptr[row], indptr[row + 1]):
        col = indices[edge]
        other = column_mate[col]
        if other < 0:
            if distance[row] + 1 != shortest:
                continue
        elif distance[other] != distance[row] + 1 or not _hk_dfs(
            other, indptr, indices, row_mate, column_mate, distance, shortest
        ):
            continue
        row_mate[row] = col
        column_mate[col] = row
        return True
    distance[row] = -1
    return False


@njit(cache=True, nogil=True)
def hk_matching(indptr, indices, column_count):
    """Return a maximum matching (row -> column, or -1) for CSR adjacency."""
    n = len(indptr) - 1
    row_mate = np.full(n, -1, np.int64)
    column_mate = np.full(column_count, -1, np.int64)
    distance = np.empty(n, np.int64)
    queue = np.empty(n, np.int64)
    total = 0
    while True:
        head, tail = 0, 0
        distance[:] = -1
        for row in range(n):
            if row_mate[row] < 0:
                distance[row] = 0
                queue[tail] = row
                tail += 1
        shortest = n + 1
        while head < tail:
            row = queue[head]
            head += 1
            if distance[row] >= shortest:
                continue
            for edge in range(indptr[row], indptr[row + 1]):
                other = column_mate[indices[edge]]
                if other < 0:
                    shortest = distance[row] + 1
                elif distance[other] < 0:
                    distance[other] = distance[row] + 1
                    queue[tail] = other
                    tail += 1
        if shortest == n + 1:
            return row_mate
        for row in range(n):
            if row_mate[row] < 0 and _hk_dfs(
                row, indptr, indices, row_mate, column_mate, distance, shortest
            ):
                total += 1
        if total == n:
            return row_mate


@njit(cache=True, nogil=True)
def auction_assignment(costs, columns, column_count, epsilon, max_steps=20_000_000):
    """Assign every row of a feasible rectangular graph using forward auction.

    Unowned columns retain zero prices. Epsilon controls cost approximation,
    not BER. A finite iteration guard raises rather than returning a partial
    assignment. Call only after checking feasibility with HK.
    """
    n, k = costs.shape
    if k == 1:
        return columns[:, 0].copy(), n
    owner = np.full(column_count, -1, np.int64)
    assigned = np.full(n, -1, np.int64)
    prices = np.zeros(column_count)
    stack = np.arange(n, dtype=np.int64)
    pending = n
    steps = 0
    while pending:
        pending -= 1
        row = stack[pending]
        best, second = np.inf, np.inf
        best_col = -1
        for t in range(k):
            col = columns[row, t]
            value = costs[row, t] + prices[col]
            if value < best:
                second, best, best_col = best, value, col
            elif value < second:
                second = value
        prices[best_col] += second - best + epsilon
        previous = owner[best_col]
        owner[best_col] = row
        assigned[row] = best_col
        if previous >= 0:
            assigned[previous] = -1
            stack[pending] = previous
            pending += 1
        steps += 1
        if steps >= max_steps:
            raise RuntimeError("Auction iteration limit reached.")
    return assigned, steps


def warmup_assignment(mode):
    """Compile kernels before cluster threads start (avoids concurrent JIT)."""
    if mode not in {"hk", "auction"}:
        return
    hk_matching(
        np.array([0, 2, 4], dtype=np.int32),
        np.array([0, 1, 0, 1], dtype=np.int32),
        2,
    )
    if mode == "auction":
        auction_assignment(
            np.array([[0., 1.], [1., 0.]]),
            np.array([[0, 1], [0, 1]], dtype=np.int64),
            2, .001,
        )
