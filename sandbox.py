from time import perf_counter

import numba as nb

def loop():
    n = 0
    for _ in range(1_000_000_000):
        n += 1
    return n

t0 = perf_counter()
loop()
print((perf_counter() - t0)*1e3, 'ms')

t0 = perf_counter()
loop()
print((perf_counter() - t0)*1e3, 'ms')


nbloop = nb.njit(nb.uint32())(loop)


t0 = perf_counter()
nbloop()
print((perf_counter() - t0)*1e3, 'ms')

t0 = perf_counter()
nbloop()
print((perf_counter() - t0)*1e3, 'ms')

t0 = perf_counter()
nbloop()
print((perf_counter() - t0)*1e3, 'ms')

t0 = perf_counter()
nbloop()
print((perf_counter() - t0)*1e3, 'ms')

t0 = perf_counter()
nbloop()
print((perf_counter() - t0)*1e3, 'ms')
