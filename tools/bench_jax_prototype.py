#!/usr/bin/env python3
"""Structural JAX benchmark: what would a lax.scan'd QTCM1 step cost?

NOT a port - a synthetic step with the real model's computational
skeleton (per step: ~13 stencil kernels of representative arithmetic
density, one masked 9-iteration 2x2 Newton, 6 zonal-filter FFT pairs,
one rfft + Thomas-scan + irfft Poisson solve), compiled as one XLA
program with lax.scan over the 72 steps of a coupling day. Numbers
bound the speedup a faithful JAX port could reach by eliminating all
Python/dispatch overhead, at the standard grid, at ~1 degree, and
vmapped over an ensemble.
"""

import time

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_enable_x64', True)


def make_step(ny, nx):
    def stencil(f, a):
        return (a * (jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1) - 2.0 * f)
                + 0.25 * (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0))
                + 0.1 * f * a)

    def newton(u, v, rhs):
        def body(_, uv):
            u, v = uv
            sv = jnp.sqrt(20.25 + u * u + v * v)
            f = 0.01 * rhs - u * (2e-5 + 3e-6 * sv)
            g = -0.01 * rhs - v * (2e-5 + 3e-6 * sv)
            det = (2e-5 + 3e-6 * sv) ** 2 + 1e-9
            return (u - f / det * 2e-5, v - g / det * 2e-5)
        return lax.fori_loop(0, 9, body, (u, v))

    def zfilter(f, fac):
        c = jnp.fft.rfft(f, axis=1)
        c = c.at[:, 1:].multiply(fac)
        return jnp.fft.irfft(c, n=nx, axis=1)

    def poisson(rhs):
        c = jnp.fft.rfft(rhs, axis=1)
        def thomas(carry, row):
            return 0.5 * carry + row, 0.5 * carry + row
        _, rows = lax.scan(thomas, jnp.zeros_like(c[0]), c)
        return jnp.fft.irfft(rows, n=nx, axis=1)

    fac = jnp.linspace(0.6, 1.0, nx // 2)[None, :]

    def step(state, _):
        a, b, u, v = state
        for _k in range(9):                    # tendency/physics kernels
            a = stencil(a, 0.3) + 1e-4 * b
            b = stencil(b, 0.2) - 1e-4 * a
        u, v = newton(u, v, a)
        for _k in range(6):                    # polar-filter applications
            a = zfilter(a, fac)
        p = poisson(b)                         # barotropic solve
        b = b + 1e-3 * p
        return (a, b, u, v), None

    return step


def bench(ny, nx, nastep, nens=None, label=''):
    step = make_step(ny, nx)
    z = jnp.ones((ny, nx), jnp.float64)
    state = (z, z * 0.5, z * 0.1, z * 0.1)

    def day(state):
        state, _ = lax.scan(step, state, None, length=nastep)
        return state

    if nens:
        state = jax.tree.map(lambda x: jnp.stack([x] * nens), state)
        day_fn = jax.jit(jax.vmap(day))
    else:
        day_fn = jax.jit(day)

    out = day_fn(state)
    jax.block_until_ready(out)                 # compile + warm
    t0 = time.time()
    n = 5
    for _ in range(n):
        out = day_fn(out)
    jax.block_until_ready(out)
    per_day = (time.time() - t0) / n
    syr = per_day * 365
    per = f' ({syr / nens:.1f} s/yr per member)' if nens else ''
    print(f'{label:<34s} {per_day * 1e3:7.1f} ms/day  = {syr:6.1f} '
          f's/sim-year{per}')


if __name__ == '__main__':
    print(f'devices: {jax.devices()}')
    bench(42, 64, 72, label='r64x42, single trajectory')
    bench(42, 64, 72, nens=16, label='r64x42, vmap 16-member ensemble')
    bench(158, 360, 432, label='~1 degree (360x158, dt=200s)')
