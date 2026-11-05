import numpy as np
from numba import njit


def neg(phi):
    return -phi


def conj(*phis):
    return np.minimum.reduce(phis)


def disj(*phis):
    return np.maximum.reduce(phis)


def slw(phi, fill, a=0, b=0):
    """
    Expects fill values that don't break the upcoming operation. E.g. we should use:
      -inf for max/eventually/until's psi
      +inf for min/always/until's phi
    """
    assert a <= b,  f"sliding window requires a <= b ({a=} > {b=} given)"

    lpad = -min(0, a)

    padded = np.pad(phi, (lpad, max(0, b)), constant_values=fill)

    lcrop = a + lpad
    rcrop = lcrop + len(phi)

    return np.lib.stride_tricks.sliding_window_view(padded, b - a + 1)[lcrop: rcrop]


def unary_temporal_op(phi, unbounded_fn, bounded_fn, fill, ab: tuple[int, int] | None = None):
    if ab is None:
        return unbounded_fn(phi[::-1], axis=0)[::-1]

    if ab == (0, 0):
        # identity
        return phi

    return bounded_fn(slw(phi, fill, *ab), axis=-1)


def always(phi, ab: tuple[int, int] | None = None):
    return unary_temporal_op(phi, np.minimum.accumulate, np.min, np.inf, ab)


def eventually(phi, ab: tuple[int, int] | None = None):
    return unary_temporal_op(phi, np.maximum.accumulate, np.max, -np.inf, ab)


@njit#(cache=True)
def _unbounded_until(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    n = len(phi)  # len(phi) == len(psi)

    out = np.empty(n, np.float32)
    out[-1] = psi[-1]

    for t in range(n - 2, -1, -1):
        out[t] = max(
            psi[t],
            min(phi[t], out[t + 1])
        )

    return out


def until(phi, psi, ab: tuple[int, int] | None = None):
    if ab is None:
        return _unbounded_until(phi, psi)

    a, b = ab

    assert 0 <= a <= b, f"Bounded until requires 0 <= a <= b ({a=} > {b=} given)"

    if a == b == 0:
        return psi

    # Need phi up to b-1, psi on [a, b]
    wphi = slw(phi, np.inf, 0, b - 1)
    wpsi = slw(psi, -np.inf, a, b)

    # prefix minima for phi
    prefix = np.full((len(wpsi), b + 1), np.inf, np.float32)
    if b > 0:
        prefix[..., 1:] = np.minimum.accumulate(
            wphi,
            axis=-1
        )

    # evaluate candidates j in [a,b]
    return np.max(
        np.minimum(
            wpsi, prefix[..., a: b + 1]),
        axis=-1
    )


if __name__ == "__main__":
    import time
    import os
    from contextlib import redirect_stdout, redirect_stderr

    import rtamt

    def run_silently(fn, *args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with (redirect_stdout(devnull), redirect_stderr(devnull)):
                return fn(*args, **kwargs)

    class _BenchTimer:
        def __init__(self, label):
            self.label = label
            self.lite_s = 0.0
            self.rtamt_s = 0.0

        def lite(self, fn, *args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.lite_s += time.perf_counter() - t0
            return out

        def rtamt(self, fn, *args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.rtamt_s += time.perf_counter() - t0
            return out

        def report(self):
            ratio = self.rtamt_s / self.lite_s if self.lite_s > 0 else float("inf")
            print(
                f"  {self.label}: stl_lite {self.lite_s * 1e3:.3f} ms, "
                f"rtamt {self.rtamt_s * 1e3:.3f} ms (rtamt/stl_lite {ratio:.1f}x)"
            )

    def _rtamt_spec():
        if hasattr(rtamt, "STLDiscreteTimeSpecification"):
            return rtamt.STLDiscreteTimeSpecification()
        return rtamt.StlDiscreteTimeSpecification()

    def _rtamt_trace(spec_str, variables, n):
        if n == 0:
            return np.array([], dtype=np.float32)
        # RTAMT 0.3.x offline interpreter requires >= 2 samples (duration unset if len(time) < 2).
        if n == 1:
            times = [0, 1]
            vars_ = {k: [v[0], v[0]] for k, v in variables.items()}
            trim = 1
        else:
            times = list(range(n))
            vars_ = {k: list(v) for k, v in variables.items()}
            trim = n
        spec = _rtamt_spec()
        for name in vars_:
            spec.declare_var(name, "float")
        spec.spec = spec_str
        spec.parse()
        data = {"time": times}
        data.update(vars_)
        return np.array(spec.evaluate(data))[:trim, 1]

    def rtamt_neg(phi):
        phi = np.asarray(phi, dtype=np.float32)
        return _rtamt_trace("not(p > 0)", {"p": phi}, len(phi))

    def rtamt_conj(phi, psi):
        phi = np.asarray(phi, dtype=np.float32)
        psi = np.asarray(psi, dtype=np.float32)
        return _rtamt_trace("(p > 0) and (q > 0)", {"p": phi, "q": psi}, len(phi))

    def rtamt_disj(phi, psi):
        phi = np.asarray(phi, dtype=np.float32)
        psi = np.asarray(psi, dtype=np.float32)
        return _rtamt_trace("(p > 0) or (q > 0)", {"p": phi, "q": psi}, len(phi))

    def rtamt_always(phi, ab=None):
        n = len(phi)
        if ab is None:
            spec_str = "always(p > 0)"
        else:
            a, b = ab
            spec_str = f"always[{a},{b}](p > 0)"
        return _rtamt_trace(spec_str, {"p": phi}, n)

    def rtamt_eventually(phi, ab=None):
        n = len(phi)
        if ab is None:
            spec_str = "eventually(p > 0)"
        else:
            a, b = ab
            spec_str = f"eventually[{a},{b}](p > 0)"
        return _rtamt_trace(spec_str, {"p": phi}, n)

    def rtamt_until(phi, psi, ab):
        n = len(phi)
        if ab is None:
            spec_str = "phi until(psi > 0)"
        else:
            a, b = ab
            spec_str = f"phi until[{a},{b}] (psi > 0)"
        return _rtamt_trace(spec_str, {"phi": phi, "psi": psi}, n)

    def assert_matches_rtamt(got, rt_full):
        if len(got) == 0:
            return
        exp = rt_full
        assert np.allclose(got, exp, equal_nan=True), (got, exp)

    def test_windows():
        phi = list(range(20))

        def check_windows(a, b):
            fill = -1
            assert fill < min(phi)

            windows = slw(phi, fill, a, b)

            w = b - a + 1
            num_wins = sum(windows.max(axis=-1) != fill)

            assert windows.shape[1] == w
            assert len(windows) == len(phi)

            assert a <= b

            if a <= b <= 0:
                assert num_wins == len(phi) - abs(b)
            elif 0 <= a <= b:
                assert num_wins == len(phi) - abs(a)
            else:  # if a <= 0 <= b:
                assert num_wins == len(phi)

            for t in range(len(phi)):
                if 0 <= t + a < len(phi):
                    assert windows[t, 0] == phi[t + a]
                else:
                    assert windows[t, 0] == fill

                if 0 <= t + b < len(phi):
                    assert windows[t, -1] == phi[t + b]
                else:
                    assert windows[t, -1] == fill

            # print(f"{a=}  {b=}  ({w=}  {num_wins=})  passed")
            # print("\n".join([str(line) for line in windows.T]), "\n")

        cases = [(-2, -1), (-2, 2), (0, 0), (1, 20), (20, 20)]
        for case in cases:
            check_windows(*case)


    def test_always():
        bench = _BenchTimer("always")

        def fn():
            P, M = 1.0, -1.0  # passing / violating robustness margins
            phi = [P, P, P, P, P]
            n = len(phi)

            # Bounded intervals with a >= 0: match RTAMT on overlapping output times
            for ab in [(0, n - 1), (0, 2), (0, 0)]:
                got = bench.lite(always, phi, ab)
                assert_matches_rtamt(got, bench.rtamt(rtamt_always, phi, ab))

            # Negative offsets: RTAMT cannot evaluate early times; check slw consistency only
            for ab in [(-2, 2), (-1, 0)]:
                windows = slw(phi, np.inf, *ab)
                test = bench.lite(always, phi, ab)
                assert np.allclose(test, np.min(windows, axis=-1))

            r = bench.lite(always, phi, (0, n - 1))
            assert r[0] >= 0

            r = bench.lite(always, [P, P, M, P, P], (0, n - 1))
            assert r[0] < 0

            r = bench.lite(always, [P, M, P], (0, 0))
            assert np.allclose(r, [P, M, P])

            assert len(bench.lite(always, phi, (0, n))) == n
            assert len(bench.lite(always, [], (0, 0))) == 0

            phi = [P, P, M, P, M]
            r = bench.lite(always, phi, None)
            assert_matches_rtamt(r, bench.rtamt(rtamt_always, phi, None))
            assert r[0] < 0

            r = bench.lite(always, [P, P, P, P], None)
            assert np.all(r > 0)

        run_silently(fn)

        bench.report()


    def test_eventually():
        bench = _BenchTimer("eventually")

        def fn():
            P, M = 1.0, -1.0

            phi = [M, M, M, M, P]
            n = len(phi)

            for ab in [(0, n - 1), (0, 2), (0, 0)]:
                got = bench.lite(eventually, phi, ab)
                assert_matches_rtamt(got, bench.rtamt(rtamt_eventually, phi, ab))

            for ab in [(-1, 0)]:
                windows = slw(phi, -np.inf, *ab)
                assert np.allclose(bench.lite(eventually, phi, ab), np.max(windows, axis=-1))

            r = bench.lite(eventually, phi, (0, n - 1))
            bench.rtamt(rtamt_eventually, phi, (0, n - 1))
            assert r[0] >= 0

            r = bench.lite(eventually, [M] * n, (0, n - 1))
            bench.rtamt(rtamt_eventually, [M] * n, (0, n - 1))
            assert r[0] < 0

            r = bench.lite(eventually, [M, P, M], (0, 0))
            bench.rtamt(rtamt_eventually, [M, P, M], (0, 0))
            assert np.allclose(r, [M, P, M])

            assert len(bench.lite(eventually, phi, (0, n))) == n
            assert len(bench.lite(eventually, [], (0, 0))) == 0

            phi = [M, M, P, M, M]
            r = bench.lite(eventually, phi, None)
            assert_matches_rtamt(r, bench.rtamt(rtamt_eventually, phi, None))

            r = bench.lite(eventually, [M, M, M, M], None)
            bench.rtamt(rtamt_eventually, [M, M, M, M], None)
            assert np.all(r < 0)

        run_silently(fn)

        bench.report()


    def test_until():
        bench = _BenchTimer("until")

        def fn():
            P, M = 1.0, -1.0

            cases = [
                ([P, P, M, P, P], [M, M, P, M, M], 0, 4),
                ([P, M, P, M, P], [M, M, M, M, P], 0, 4),
                ([M, M, M], [M, M, M], 0, 2),
                ([P, P, P], [M, M, M], 0, 2),
                ([P, P, P, M, M], [M, M, P, M, M], 0, 2),
                ([P, M, P], [P, M, M], 1, 1),
            ]
            for phi, psi, a, b in cases:
                got = bench.lite(until, phi, psi, (a, b))
                assert_matches_rtamt(got, bench.rtamt(rtamt_until, phi, psi, (a, b)))
                got = bench.lite(until, phi, psi, None)
                assert_matches_rtamt(got, bench.rtamt(rtamt_until, phi, psi, None))

            # a = b = 0: returns psi unchanged
            psi = [P, M, P]
            r = bench.lite(until, [M, P, M], psi, (0, 0))
            assert np.array_equal(r, psi)

            # full-episode: phi holds until psi releases inside horizon
            r = bench.lite(until, [P, P, M, P, P], [M, M, P, M, M], (0, 4))
            assert r.shape == (5,)
            assert r[0] >= 0

            # full-episode: psi only at end but phi violated earlier
            r = bench.lite(until, [P, M, P, M, P], [M, M, M, M, P], (0, 4))
            assert r[0] < 0

            # psi never reaches +1 within horizon
            r = bench.lite(until, [P, P, P], [M, M, M], (0, 2))
            assert np.all(r < 0)

            # release at offset a only (j = a)
            r = bench.lite(until, [P, M, P], [M, P, M], (1, 1))
            assert r[0] > 0
            assert r[1] < 0

            try:
                bench.lite(until, [P, M], [M, P], (2, 1))
                raise RuntimeError("expected assert for a > b")
            except AssertionError:
                pass

        run_silently(fn)

        bench.report()


    def test_neg():
        bench = _BenchTimer("neg")

        def fn():
            P, M = 1.0, -1.0
            cases = [
                [P, P, M, P, P],
                [M, M, M, M, M],
                [P, M, P, M, P],
                [M],
                [],
            ]
            for phi in cases:
                phi = np.asarray(phi, dtype=np.float32)
                got = bench.lite(neg, phi)
                if len(phi) == 0:
                    assert len(got) == 0
                    continue
                assert_matches_rtamt(got, bench.rtamt(rtamt_neg, phi))

            phi = np.asarray([P, M, P], dtype=np.float32)
            assert np.allclose(bench.lite(neg, phi), [-P, -M, -P])

        run_silently(fn)
        bench.report()


    def test_conj():
        bench = _BenchTimer("conj")

        def fn():
            P, M = 1.0, -1.0
            cases = [
                ([P, P, M, P, P], [M, M, P, M, M]),
                ([P, M, P, M, P], [M, M, M, M, P]),
                ([P, P, P], [M, M, M]),
                ([P], [M]),
            ]
            for phi, psi in cases:
                phi = np.asarray(phi, dtype=np.float32)
                psi = np.asarray(psi, dtype=np.float32)
                got = bench.lite(conj, phi, psi)
                assert_matches_rtamt(got, bench.rtamt(rtamt_conj, phi, psi))

            phi = np.asarray([P, M, P], dtype=np.float32)
            psi = np.asarray([M, P, M], dtype=np.float32)
            assert np.allclose(bench.lite(conj, phi, psi), [M, M, M])

        run_silently(fn)
        bench.report()


    def test_disj():
        bench = _BenchTimer("disj")

        def fn():
            P, M = 1.0, -1.0
            cases = [
                ([P, P, M, P, P], [M, M, P, M, M]),
                ([P, M, P, M, P], [M, M, M, M, P]),
                ([M, M, M], [M, M, M]),
                ([M], [P]),
            ]
            for phi, psi in cases:
                phi = np.asarray(phi, dtype=np.float32)
                psi = np.asarray(psi, dtype=np.float32)
                got = bench.lite(disj, phi, psi)
                assert_matches_rtamt(got, bench.rtamt(rtamt_disj, phi, psi))

            phi = np.asarray([P, M, P], dtype=np.float32)
            psi = np.asarray([M, P, M], dtype=np.float32)
            assert np.allclose(bench.lite(disj, phi, psi), [P, P, P])

        run_silently(fn)
        bench.report()

    def benchmark_episode_scale(n_ep=100, n_steps=360, ab=(0, 10)):
        """Many episodes, fixed formula: hoisted RTAMT (parse once) vs stl_lite."""
        P, M = 1.0, -1.0
        if ab is not None:
            a, b = ab
        else:
            a = b = None
        rng = np.random.default_rng(0)
        traces = [
            np.where(rng.random(n_steps) > 0.3, P, M).astype(np.float32)
            for _ in range(n_ep)
        ]
        phi_traces = traces
        psi_traces = [
            np.where(rng.random(n_steps) > 0.7, P, M).astype(np.float32)
            for _ in range(n_ep)
        ]
        times = list(range(n_steps))

        def _hoisted_spec(spec_str, var_names):
            spec = _rtamt_spec()
            for name in var_names:
                spec.declare_var(name, "float")
            spec.spec = spec_str
            t0 = time.perf_counter()
            spec.parse()
            parse_ms = (time.perf_counter() - t0) * 1e3
            return spec, parse_ms

        # warmup RTAMT / ANTLR before any timed rtamt work
        run_silently(rtamt_always ,traces[0], ab)

        print(
            f"episode-scale ({n_ep} episodes, {n_steps} steps, ab={ab}):"
        )

        def compare_always():
            # --- always[a,b] ---
            t0 = time.perf_counter()
            for phi in traces:
                always(phi, ab)
            lite_always_ms = (time.perf_counter() - t0) * 1e3

            if ab is not None:
                spec, parse_ms = _hoisted_spec(f"always[{a},{b}](p > 0)", ["p"])
            else:
                spec, parse_ms = _hoisted_spec(f"always(p > 0)", ["p"])

            t0 = time.perf_counter()
            for phi in traces:
                spec.evaluate({"time": times, "p": phi.tolist()})
            hoisted_always_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            for phi in traces:
                rtamt_always(phi, ab)
            naive_always_ms = (time.perf_counter() - t0) * 1e3

            return (
                f"  always[{a},{b}]: parse-once {parse_ms:.3f} ms | "
                f"stl_lite {lite_always_ms:.1f} ms ({lite_always_ms / n_ep:.3f} ms/ep) | "
                f"rtamt hoisted {hoisted_always_ms:.1f} ms ({hoisted_always_ms / n_ep:.3f} ms/ep) | "
                f"rtamt naive {naive_always_ms:.1f} ms ({naive_always_ms / n_ep:.3f} ms/ep) | "
                f"stl_lite vs hoisted {hoisted_always_ms / lite_always_ms:.1f}x"
            )

        print(run_silently(compare_always))

        def compare_eventually():
            # --- eventually[a,b] ---
            t0 = time.perf_counter()
            for phi in traces:
                eventually(phi, ab)
            lite_ev_ms = (time.perf_counter() - t0) * 1e3

            if ab is not None:
                spec_ev, parse_ev_ms = _hoisted_spec(f"eventually[{a},{b}](p > 0)", ["p"])
            else:
                spec_ev, parse_ev_ms = _hoisted_spec(f"eventually(p > 0)", ["p"])
            t0 = time.perf_counter()
            for phi in traces:
                spec_ev.evaluate({"time": times, "p": phi.tolist()})
            hoisted_ev_ms = (time.perf_counter() - t0) * 1e3

            return (
                f"  eventually[{a},{b}]: parse-once {parse_ev_ms:.3f} ms | "
                f"stl_lite {lite_ev_ms:.1f} ms ({lite_ev_ms / n_ep:.3f} ms/ep) | "
                f"rtamt hoisted {hoisted_ev_ms:.1f} ms ({hoisted_ev_ms / n_ep:.3f} ms/ep) | "
                f"stl_lite vs hoisted {hoisted_ev_ms / lite_ev_ms:.1f}x"
            )

        print(run_silently(compare_eventually))

        def compare_until():
            # --- until[a,b] ---
            t0 = time.perf_counter()
            for phi, psi in zip(phi_traces, psi_traces):
                until(phi, psi, ab)
            lite_until_ms = (time.perf_counter() - t0) * 1e3

            if ab is not None:
                spec_u, parse_u_ms = _hoisted_spec(
                    f"phi until[{a},{b}] (psi > 0)", ["phi", "psi"]
                )
            else:
                spec_u, parse_u_ms = _hoisted_spec(
                    f"phi until(psi > 0)", ["phi", "psi"]
                )

            t0 = time.perf_counter()
            for phi, psi in zip(phi_traces, psi_traces):
                spec_u.evaluate(
                    {"time": times, "phi": phi.tolist(), "psi": psi.tolist()}
                )
            hoisted_until_ms = (time.perf_counter() - t0) * 1e3

            return (
                f"  until[{a},{b}]: parse-once {parse_u_ms:.3f} ms | "
                f"stl_lite {lite_until_ms:.1f} ms ({lite_until_ms / n_ep:.3f} ms/ep) | "
                f"rtamt hoisted {hoisted_until_ms:.1f} ms ({hoisted_until_ms / n_ep:.3f} ms/ep) | "
                f"stl_lite vs hoisted {hoisted_until_ms / lite_until_ms:.1f}x"
            )

        print(run_silently(compare_until))

        def compare_neg():
            t0 = time.perf_counter()
            for phi in traces:
                neg(phi)
            lite_ms = (time.perf_counter() - t0) * 1e3

            spec, parse_ms = _hoisted_spec("not(p > 0)", ["p"])
            t0 = time.perf_counter()
            for phi in traces:
                spec.evaluate({"time": times, "p": phi.tolist()})
            hoisted_ms = (time.perf_counter() - t0) * 1e3

            return (
                f"  neg: parse-once {parse_ms:.3f} ms | "
                f"stl_lite {lite_ms:.1f} ms ({lite_ms / n_ep:.3f} ms/ep) | "
                f"rtamt hoisted {hoisted_ms:.1f} ms ({hoisted_ms / n_ep:.3f} ms/ep) | "
                f"stl_lite vs hoisted {hoisted_ms / lite_ms:.1f}x"
            )

        def compare_conj():
            t0 = time.perf_counter()
            for phi, psi in zip(phi_traces, psi_traces):
                conj(phi, psi)
            lite_ms = (time.perf_counter() - t0) * 1e3

            spec, parse_ms = _hoisted_spec("(p > 0) and (q > 0)", ["p", "q"])
            t0 = time.perf_counter()
            for phi, psi in zip(phi_traces, psi_traces):
                spec.evaluate(
                    {"time": times, "p": phi.tolist(), "q": psi.tolist()}
                )
            hoisted_ms = (time.perf_counter() - t0) * 1e3

            return (
                f"  conj: parse-once {parse_ms:.3f} ms | "
                f"stl_lite {lite_ms:.1f} ms ({lite_ms / n_ep:.3f} ms/ep) | "
                f"rtamt hoisted {hoisted_ms:.1f} ms ({hoisted_ms / n_ep:.3f} ms/ep) | "
                f"stl_lite vs hoisted {hoisted_ms / lite_ms:.1f}x"
            )

        def compare_disj():
            t0 = time.perf_counter()
            for phi, psi in zip(phi_traces, psi_traces):
                disj(phi, psi)
            lite_ms = (time.perf_counter() - t0) * 1e3

            spec, parse_ms = _hoisted_spec("(p > 0) or (q > 0)", ["p", "q"])
            t0 = time.perf_counter()
            for phi, psi in zip(phi_traces, psi_traces):
                spec.evaluate(
                    {"time": times, "p": phi.tolist(), "q": psi.tolist()}
                )
            hoisted_ms = (time.perf_counter() - t0) * 1e3

            return (
                f"  disj: parse-once {parse_ms:.3f} ms | "
                f"stl_lite {lite_ms:.1f} ms ({lite_ms / n_ep:.3f} ms/ep) | "
                f"rtamt hoisted {hoisted_ms:.1f} ms ({hoisted_ms / n_ep:.3f} ms/ep) | "
                f"stl_lite vs hoisted {hoisted_ms / lite_ms:.1f}x"
            )

        print(run_silently(compare_neg))
        print(run_silently(compare_conj))
        print(run_silently(compare_disj))

    until(np.zeros(50, dtype=np.float32), np.zeros(50, dtype=np.float32))  # pre-compile

    print("timing (total per test group):")
    test_windows()
    test_always()
    test_eventually()
    test_until()
    test_neg()
    test_conj()
    test_disj()
    print()
    benchmark_episode_scale()
    benchmark_episode_scale(ab=None)
    print("stl_lite tests passed")
