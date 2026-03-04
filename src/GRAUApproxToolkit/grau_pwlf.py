import numpy as np
import matplotlib.pyplot as plt

class GRAU_PWLF:
    """
    Greedy PWLF with optional slope quantization.

    Slope modes (slopes(mode)):
      - "original"   -> float slopes
      - "PoT"        -> float slopes quantized to nearest power-of-two
      - "APoT"       -> float slopes quantized to sum of DISTINCT powers-of-two (each exponent used at most once)
      - "PoT-INT"    -> int exponents for PoT slopes (e.g., 2^-10 -> -10)
      - "APoT-INT"   -> object array of sorted exponent lists for APoT slopes (e.g., [-10,-2,4])
      - "PoT-SIGN"   -> int sign (-1,0,+1) for PoT slopes
      - "APoT-SIGN"  -> int sign (-1,0,+1) for APoT slopes

    Plot/eval modes:
      - plot_final(mode=...), __call__(x, mode=...) accept: "original", "PoT", "APoT"
      - plot_steps() has NO parameters and plots only original PWLF steps.
      - plot_compare(title_name) plots 4 panels:
          original nonlinear, PWLF, PoT-PWLF, APoT-PWLF
    """

    def __init__(
        self,
        x, y,
        segment_num=8,
        require_integer=True,
        sort_by_x=True,
        min_improvement=0.0,
        min_gap=1,
        default_figsize=(10, 5),
        pot_min_exp=-10,
        pot_max_exp=6,
    ):
        self._qcache = {}

        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape != y.shape:
            raise ValueError("x and y must be 1D arrays with the same length.")
        if len(x) < 3:
            raise ValueError("At least 3 points are required.")
        if segment_num < 1:
            raise ValueError("segment_num must be >= 1.")
        if min_gap < 0:
            raise ValueError("min_gap must be >= 0.")

        self._order = np.arange(len(x))
        if sort_by_x:
            self._order = np.argsort(x)
            x = x[self._order]
            y = y[self._order]

        self.x = x
        self.y = y

        self.require_integer = bool(require_integer)
        self.min_improvement = float(min_improvement)
        self.min_gap = int(min_gap)

        self._target_segments = int(segment_num)
        self._target_breaks = max(0, self._target_segments - 1)

        self._figsize = (float(default_figsize[0]), float(default_figsize[1]))

        self._pot_min_exp = int(pot_min_exp)
        self._pot_max_exp = int(pot_max_exp)
        if self._pot_min_exp > self._pot_max_exp:
            raise ValueError("pot_min_exp must be <= pot_max_exp.")

        self._breaks_x = []
        self._breaks_history = []

        self._candidate_cache = {}

        self._fit_incremental()
        self._rebuild_pwlf()

    # ---------- settings ----------
    def figsize(self, size):
        if not (isinstance(size, (tuple, list)) and len(size) == 2):
            raise ValueError("size must be a tuple/list like (width, height).")
        self._figsize = (float(size[0]), float(size[1]))
        return self

    def pot_range(self, min_exp=-10, max_exp=6):
        min_exp = int(min_exp)
        max_exp = int(max_exp)
        if min_exp > max_exp:
            raise ValueError("min_exp must be <= max_exp.")
        self._pot_min_exp = min_exp
        self._pot_max_exp = max_exp
        self._qcache.clear()
        return self

    # ---------- properties ----------
    @property
    def breakpoints(self):
        return np.array(self._breaks_x, dtype=int)

    @property
    def segment_num(self):
        return int(len(self._slopes))

    @property
    def knots_x(self):
        return self._knots_x.copy()

    @property
    def knots_y(self):
        return self._knots_y.copy()

    @property
    def snapshots(self):
        return [s.copy() for s in self._breaks_history]

    # ---------- helpers ----------
    def _validate_plot_eval_mode(self, mode):
        mode = str(mode)
        if mode not in ("original", "PoT", "APoT"):
            raise ValueError('mode must be one of: "original", "PoT", "APoT".')
        return mode

    def _validate_slope_mode(self, mode):
        mode = str(mode)
        allowed = ("original", "PoT", "APoT", "PoT-INT", "APoT-INT", "PoT-SIGN", "APoT-SIGN")
        if mode not in allowed:
            raise ValueError(f'mode must be one of: {allowed}.')
        return mode

    # ---------- PoT/APoT ----------        
    def _nearest_power_of_two_limited(self, val):
        if val == 0:
            return 0.0, 0, 0
        sign = int(np.sign(val))
        a = abs(val)
    
        exp = int(np.round(np.log2(a)))
            
        # If exponent is below min_exp, choose the closer one between 0 and 2^min_exp.
        # This avoids forcing extremely small slopes to 2^min_exp (can blow up intercept),
        # while still keeping better approximation for values near 2^min_exp.
        if exp < self._pot_min_exp:
            min_mag = 2.0 ** self._pot_min_exp
            # Compare |val - 0| vs |val - min_mag|
            # min_mag is closer if |val| >= 0.5 * min_mag
            if a >= 0.5 * min_mag:
                return float(sign * min_mag), int(self._pot_min_exp), sign
            else:
                return 0.0, 0, 0
    
        # If exponent is above max_exp, clamp to max_exp (too large slope -> saturate to max representable).
        if exp > self._pot_max_exp:
            exp = self._pot_max_exp
    
        return float(sign * (2.0 ** exp)), int(exp), sign

    def _nearest_additive_power_of_two_limited(self, val):
        if val == 0:
            return 0.0, [], 0
    
        sign = int(np.sign(val))
        remaining = abs(val)

        min_mag = 2.0 ** self._pot_min_exp

        # If magnitude is smaller than the minimum representable power,
        # choose the closer one between 0 and (sign * 2^min_exp).
        if remaining < min_mag:
            # min_mag is closer if |val| >= 0.5 * min_mag
            if remaining >= 0.5 * min_mag:
                return float(sign * min_mag), [int(self._pot_min_exp)], sign
            else:
                return 0.0, [], 0
    
        chosen_exps_desc = []
        result = 0.0
    
        for e in range(self._pot_max_exp, self._pot_min_exp - 1, -1):
            p = 2.0 ** e
            if remaining >= p:
                result += p
                remaining -= p
                chosen_exps_desc.append(int(e))
            if remaining < min_mag:
                break

        exps_sorted = sorted(chosen_exps_desc)
        return float(sign * result), exps_sorted, sign

    def _ensure_quantized(self, mode):
        mode = self._validate_plot_eval_mode(mode)
        if mode == "original":
            return
        if mode in self._qcache:
            return

        bx = self._knots_x
        by = self._knots_y

        if mode == "PoT":
            mq = np.empty_like(self._slopes, dtype=float)
            exp = np.empty_like(self._slopes, dtype=int)
            sgn = np.empty_like(self._slopes, dtype=int)

            for i, v in enumerate(self._slopes):
                qv, qe, qs = self._nearest_power_of_two_limited(v)
                mq[i] = qv
                exp[i] = qe
                sgn[i] = qs

            bq = np.empty_like(mq, dtype=float)
            for i in range(len(mq)):
                bq[i] = by[i] - mq[i] * bx[i]

            self._qcache["PoT"] = {"m": mq, "b": bq, "exp": exp, "sign": sgn}

        elif mode == "APoT":
            mq = np.empty_like(self._slopes, dtype=float)
            exps_obj = np.empty(len(self._slopes), dtype=object)
            sgn = np.empty_like(self._slopes, dtype=int)

            for i, v in enumerate(self._slopes):
                qv, exps_sorted, qs = self._nearest_additive_power_of_two_limited(v)
                mq[i] = qv
                exps_obj[i] = list(exps_sorted)
                sgn[i] = qs

            bq = np.empty_like(mq, dtype=float)
            for i in range(len(mq)):
                bq[i] = by[i] - mq[i] * bx[i]

            self._qcache["APoT"] = {"m": mq, "b": bq, "exps": exps_obj, "sign": sgn}

    # ---------- API: slopes/intercepts ----------
    def slopes(self, mode="original"):
        mode = self._validate_slope_mode(mode)

        if mode == "original":
            return self._slopes.copy()

        if mode in ("PoT", "PoT-INT", "PoT-SIGN"):
            self._ensure_quantized("PoT")
            if mode == "PoT":
                return self._qcache["PoT"]["m"].copy()
            if mode == "PoT-INT":
                return self._qcache["PoT"]["exp"].copy()
            return self._qcache["PoT"]["sign"].copy()

        self._ensure_quantized("APoT")
        if mode == "APoT":
            return self._qcache["APoT"]["m"].copy()
        if mode == "APoT-INT":
            return self._qcache["APoT"]["exps"].copy()
        return self._qcache["APoT"]["sign"].copy()

    def intercepts(self, mode="original"):
        mode = str(mode)
        if mode not in ("original", "PoT", "APoT"):
            raise ValueError('intercepts(mode) supports only: "original","PoT","APoT".')

        if mode == "original":
            return self._intercepts.copy()

        self._ensure_quantized(mode)
        return self._qcache[mode]["b"].copy()

    # ---------- interpolation ----------
    def _interp_y(self, xq):
        xq = np.asarray(xq, dtype=float)
        yq = np.interp(xq, self.x, self.y)

        x0, y0 = self.x[0], self.y[0]
        x1, y1 = self.x[-1], self.y[-1]

        left = xq < x0
        if np.any(left):
            dx = self.x[1] - self.x[0]
            m = (self.y[1] - self.y[0]) / dx if dx != 0 else 0.0
            yq[left] = y0 + m * (xq[left] - x0)

        right = xq > x1
        if np.any(right):
            dx = self.x[-1] - self.x[-2]
            m = (self.y[-1] - self.y[-2]) / dx if dx != 0 else 0.0
            yq[right] = y1 + m * (xq[right] - x1)

        return yq

    # ---------- geometry ----------
    @staticmethod
    def _line_params(x0, y0, x1, y1):
        A = y0 - y1
        B = x1 - x0
        C = x0 * y1 - x1 * y0
        denom = np.sqrt(A * A + B * B)
        return A, B, C, denom

    def _perp_dist(self, x_pts, y_pts, x0, y0, x1, y1):
        A, B, C, denom = self._line_params(x0, y0, x1, y1)
        if denom == 0:
            return np.zeros_like(x_pts, dtype=float)
        return np.abs(A * x_pts + B * y_pts + C) / denom

    # ---------- candidates ----------
    def _compute_top1_candidate(self, xl, xr):
        if xr <= xl:
            return None

        yl = float(self._interp_y(np.array([xl]))[0])
        yr = float(self._interp_y(np.array([xr]))[0])

        mask = (self.x > xl) & (self.x < xr)
        if not np.any(mask):
            return None

        xs = self.x[mask]
        ys = self.y[mask]
        d = self._perp_dist(xs, ys, xl, yl, xr, yr)
        if d.size == 0:
            return None

        k = int(np.argmax(d))
        dist_best = float(d[k])
        if dist_best <= self.min_improvement:
            return None

        x_raw = float(xs[k])
        xb = int(np.rint(x_raw)) if self.require_integer else int(x_raw)
        if not (xl < xb < xr):
            return None

        return (dist_best, xb)

    def _is_valid_breakpoint(self, xb, used_set):
        for u in used_set:
            if abs(int(xb) - int(u)) <= self.min_gap:
                return False
        return True

    # ---------- incremental fit ----------
    def _fit_incremental(self):
        xl0 = float(self.x[0])
        xr0 = float(self.x[-1])

        segments = [(xl0, xr0)]
        self._candidate_cache = {(xl0, xr0): self._compute_top1_candidate(xl0, xr0)}

        self._breaks_x = []
        self._breaks_history = [np.array(self._breaks_x, dtype=int)]

        for _ in range(self._target_breaks):
            used = set(self._breaks_x)
            used.add(int(np.rint(xl0)) if self.require_integer else int(xl0))
            used.add(int(np.rint(xr0)) if self.require_integer else int(xr0))

            per_seg = []
            for seg in segments:
                cand = self._candidate_cache.get(seg, None)
                if cand is None:
                    continue
                dist_val, xb = cand
                per_seg.append((dist_val, xb, seg))

            if not per_seg:
                break

            per_seg.sort(key=lambda t: -t[0])

            chosen = None
            for dist_val, xb, seg in per_seg:
                if self._is_valid_breakpoint(xb, used):
                    chosen = (dist_val, xb, seg)
                    break

            if chosen is None:
                break

            _, xb, seg_to_split = chosen
            if xb in self._breaks_x:
                break

            self._breaks_x.append(int(xb))
            self._breaks_x = sorted(set(self._breaks_x))
            self._breaks_history.append(np.array(self._breaks_x, dtype=int))

            xl, xr = seg_to_split
            left_seg = (xl, float(xb))
            right_seg = (float(xb), xr)

            segments.remove(seg_to_split)
            segments.append(left_seg)
            segments.append(right_seg)

            self._candidate_cache.pop(seg_to_split, None)
            self._candidate_cache[left_seg] = self._compute_top1_candidate(*left_seg)
            self._candidate_cache[right_seg] = self._compute_top1_candidate(*right_seg)

    # ---------- build final pwlf ----------
    def _rebuild_pwlf(self):
        xl0 = float(self.x[0])
        xr0 = float(self.x[-1])

        self._knots_x = np.array([xl0] + sorted(self._breaks_x) + [xr0], dtype=float)
        self._knots_y = self._interp_y(self._knots_x).astype(float)

        slopes = []
        intercepts = []
        for i in range(len(self._knots_x) - 1):
            x0, y0 = self._knots_x[i], self._knots_y[i]
            x1, y1 = self._knots_x[i + 1], self._knots_y[i + 1]
            if x1 == x0:
                slopes.append(np.inf)
                intercepts.append(np.nan)
            else:
                m = (y1 - y0) / (x1 - x0)
                b = y0 - m * x0
                slopes.append(m)
                intercepts.append(b)

        self._slopes = np.array(slopes, dtype=float)
        self._intercepts = np.array(intercepts, dtype=float)
        self._qcache.clear()

    # ---------- evaluation ----------
    def __call__(self, x_in, mode="original"):
        mode = self._validate_plot_eval_mode(mode)

        if mode == "original":
            m = self._slopes
            b = self._intercepts
        else:
            self._ensure_quantized(mode)
            m = self._qcache[mode]["m"]
            b = self._qcache[mode]["b"]

        xq = np.asarray(x_in, dtype=float)
        scalar = (xq.ndim == 0)
        xq_flat = xq.reshape(-1)

        bx = self._knots_x
        seg = np.searchsorted(bx, xq_flat, side="right") - 1
        seg = np.where(seg < 0, 0, seg)
        seg = np.where(seg >= len(m), len(m) - 1, seg)

        yq = m[seg] * xq_flat + b[seg]
        yq = yq.reshape(xq.shape)
        return float(yq) if scalar else yq

    # ---------- plotting ----------
    def plot_final(self, mode="original", title=None):
        mode = self._validate_plot_eval_mode(mode)

        if title is None:
            title = f"Final PWLF ({mode}) | segments={self.segment_num} | internal_breaks={len(self._breaks_x)}"

        if mode == "original":
            m = self._slopes
            b = self._intercepts
        else:
            self._ensure_quantized(mode)
            m = self._qcache[mode]["m"]
            b = self._qcache[mode]["b"]

        plt.figure(figsize=self._figsize)
        plt.plot(self.x, self.y, linewidth=1, label="original")

        for i in range(len(self._knots_x) - 1):
            x0 = self._knots_x[i]
            x1 = self._knots_x[i + 1]
            y0p = m[i] * x0 + b[i]
            y1p = m[i] * x1 + b[i]
            plt.plot([x0, x1], [y0p, y1p], linewidth=2)

        if len(self._breaks_x) > 0:
            bx_pts = np.array(self._breaks_x, dtype=float)
            by_pts = self.__call__(bx_pts, mode=mode)
            plt.scatter(bx_pts, by_pts, s=50, marker="x", label="breakpoints")

        plt.grid(True)
        plt.legend(loc="best")
        plt.title(title)
        plt.show()

    def plot_steps(self):
        xl0 = float(self.x[0])
        xr0 = float(self.x[-1])

        for k, breaks_k in enumerate(self._breaks_history):
            knots_x = np.array([xl0] + sorted(map(int, breaks_k.tolist())) + [xr0], dtype=float)
            knots_y = self._interp_y(knots_x).astype(float)

            m = []
            b = []
            for i in range(len(knots_x) - 1):
                x0, y0 = knots_x[i], knots_y[i]
                x1, y1 = knots_x[i + 1], knots_y[i + 1]
                if x1 == x0:
                    mm = np.inf
                    bb = np.nan
                else:
                    mm = (y1 - y0) / (x1 - x0)
                    bb = y0 - mm * x0
                m.append(mm)
                b.append(bb)
            m = np.array(m, dtype=float)
            b = np.array(b, dtype=float)

            plt.figure(figsize=self._figsize)
            plt.plot(self.x, self.y, linewidth=1, label="original")

            for i in range(len(knots_x) - 1):
                x0, x1 = knots_x[i], knots_x[i + 1]
                y0p = m[i] * x0 + b[i]
                y1p = m[i] * x1 + b[i]
                plt.plot([x0, x1], [y0p, y1p], linewidth=2)

            if len(breaks_k) > 0:
                bx_pts = breaks_k.astype(float)
                seg = np.searchsorted(knots_x, bx_pts, side="right") - 1
                seg = np.clip(seg, 0, len(m) - 1)
                by_pts = m[seg] * bx_pts + b[seg]
                plt.scatter(bx_pts, by_pts, s=50, marker="x", label="breakpoints")

            plt.grid(True)
            plt.legend(loc="best")
            plt.title(f"Step {k} | ORIGINAL PWLF | segments={len(knots_x)-1} | breaks={breaks_k}")
            plt.show()

    def plot_compare(self, title_name):
        title_name = str(title_name)

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        axes[0].plot(self.x, self.y, linewidth=1)
        axes[0].grid(True)
        axes[0].set_title(f"Original {title_name}")

        def _plot_pwlf(ax, mode, title):
            if mode == "original":
                m = self._slopes
                b = self._intercepts
            else:
                self._ensure_quantized(mode)
                m = self._qcache[mode]["m"]
                b = self._qcache[mode]["b"]

            ax.plot(self.x, self.y, linewidth=1, label="original")
            for i in range(len(self._knots_x) - 1):
                x0 = self._knots_x[i]
                x1 = self._knots_x[i + 1]
                y0p = m[i] * x0 + b[i]
                y1p = m[i] * x1 + b[i]
                ax.plot([x0, x1], [y0p, y1p], linewidth=2)
            ax.grid(True)
            ax.set_title(title)

        _plot_pwlf(axes[1], "original", f"Piecewise Linear Fitted (PWLF) {title_name}")
        _plot_pwlf(axes[2], "PoT", f"Power-of-Two (PoT) Approximated PWLF {title_name}")
        _plot_pwlf(axes[3], "APoT", f"Additive Power-of-Two (APoT) Approximated PWLF {title_name}")

        plt.tight_layout()
        plt.show()