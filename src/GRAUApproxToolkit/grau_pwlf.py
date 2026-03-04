# grau_pwlf.py
import numpy as np
import matplotlib.pyplot as plt


class GRAU_PWLF:
    """
    Greedy PWLF with optional slope quantization.

    -------------------------------------------------------------------------
    Overview
    -------------------------------------------------------------------------
    This class takes sampled points (x, y) of a nonlinear function and builds
    a piecewise-linear function (PWLF) using a greedy breakpoint selection.
    Breakpoints are optionally forced to be integers, and a minimum distance
    between breakpoints can be enforced.

    After the final PWLF is built, we can derive slope-quantized variants:
      - PoT  : each slope is approximated by a (signed) power-of-two
      - APoT : each slope is approximated by a (signed) SUM of DISTINCT powers-of-two
              (each exponent is used at most once)

    Important design choice:
      - Quantization happens AFTER the final PWLF is built.
      - We keep the knots_x fixed and recompute intercepts such that the piecewise
        line remains continuous at the left knot of each segment:
            b_i = y_knot[i] - m_i * x_knot[i]

    -------------------------------------------------------------------------
    Slope modes (slopes(mode))
    -------------------------------------------------------------------------
      - "original"   -> float slopes of fitted PWLF
      - "PoT"        -> float slopes quantized to nearest representable PoT value
      - "APoT"       -> float slopes quantized to nearest representable APoT value
      - "PoT-INT"    -> integer exponent e for PoT slopes (2^e). 0 is used as placeholder for slope=0
      - "APoT-INT"   -> object array of sorted exponent lists for APoT slopes (e.g., [-10,-2,4])
      - "PoT-SIGN"   -> int sign (-1,0,+1) for PoT slopes
      - "APoT-SIGN"  -> int sign (-1,0,+1) for APoT slopes

    NOTE:
      - INT modes are ALWAYS consistent with the float slopes because they are stored
        from the same quantizer decision (no recomputation).

    -------------------------------------------------------------------------
    Plot/eval modes
    -------------------------------------------------------------------------
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
        # Cache for quantized variants. Must exist before _rebuild_pwlf().
        # Format:
        #   self._qcache["PoT"]  = {"m":..., "b":..., "exp":..., "sign":...}
        #   self._qcache["APoT"] = {"m":..., "b":..., "exps":..., "sign":...}
        self._qcache = {}

        # -------- sanitize inputs --------
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

        # -------- optional sort by x --------
        self._order = np.arange(len(x))
        if sort_by_x:
            self._order = np.argsort(x)
            x = x[self._order]
            y = y[self._order]

        self.x = x
        self.y = y

        # -------- settings --------
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

        # -------- internal states for greedy fitting --------
        self._breaks_x = []           # integer breakpoints (internal breaks, no endpoints)
        self._breaks_history = []     # snapshots after each added breakpoint
        self._candidate_cache = {}    # (xl, xr) -> top1 candidate

        # Build PWLF
        self._fit_incremental()
        self._rebuild_pwlf()

    # =====================================================================
    # Settings
    # =====================================================================
    def figsize(self, size):
        """
        Set figure size for plotting.
        """
        if not (isinstance(size, (tuple, list)) and len(size) == 2):
            raise ValueError("size must be a tuple/list like (width, height).")
        self._figsize = (float(size[0]), float(size[1]))
        return self

    def pot_range(self, min_exp=-10, max_exp=6):
        """
        Set exponent range for PoT/APoT quantization and clear quantization cache.
        """
        min_exp = int(min_exp)
        max_exp = int(max_exp)
        if min_exp > max_exp:
            raise ValueError("min_exp must be <= max_exp.")
        self._pot_min_exp = min_exp
        self._pot_max_exp = max_exp
        self._qcache.clear()
        return self

    # =====================================================================
    # Properties
    # =====================================================================
    @property
    def breakpoints(self):
        """
        Internal breakpoints (integers), excluding endpoints.
        """
        return np.array(self._breaks_x, dtype=int)

    @property
    def segment_num(self):
        """
        Current number of segments in final PWLF.
        """
        return int(len(self._slopes))

    @property
    def knots_x(self):
        """
        Knot x coordinates including endpoints: [x0, b1, b2, ..., xN]
        """
        return self._knots_x.copy()

    @property
    def knots_y(self):
        """
        Nonlinear function y values on knots_x (interpolated from samples).
        """
        return self._knots_y.copy()

    @property
    def snapshots(self):
        """
        A list of breakpoint snapshots during greedy fitting.
        Each entry is an int array of internal breakpoints.
        """
        return [s.copy() for s in self._breaks_history]

    # =====================================================================
    # Mode validation
    # =====================================================================
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

    # =====================================================================
    # PoT/APoT quantizers
    # =====================================================================
    def _quantize_pot(self, val):
        """
        Quantize slope to the nearest representable PoT value within exponent range.

        Representable set:
          {0} U { +/- 2^e | e in [min_exp, max_exp] }

        IMPORTANT FIX:
          - Clamp BOTH e_lo and e_hi into [min_exp, max_exp].
          - Only use log2-neighbor candidates when a is strictly inside (min_mag, max_mag).
            For a <= min_mag, candidates should be {0, min_mag}.
            For a >= max_mag, candidates should include max_mag (and also {0, min_mag}).

        Return:
          (quantized_float, exponent_int, sign_int)
        """
        if val == 0:
            return 0.0, 0, 0

        sign = int(np.sign(val))
        a = abs(val)

        min_e = self._pot_min_exp
        max_e = self._pot_max_exp
        min_mag = 2.0 ** min_e
        max_mag = 2.0 ** max_e

        # Candidate list: (mag, exp, sign_for_output)
        cands = []

        # Always allow 0 candidate
        cands.append((0.0, 0, 0))

        # Always allow min magnitude candidate (smallest non-zero representable)
        cands.append((min_mag, min_e, sign))

        # If too large, allow max magnitude candidate
        if a >= max_mag:
            cands.append((max_mag, max_e, sign))

        # Only if strictly inside (min_mag, max_mag), add log2 neighbors
        elif a > min_mag:
            # a is in (min_mag, max_mag)
            e_lo = int(np.floor(np.log2(a)))
            e_hi = e_lo + 1

            # HARD clamp BOTH sides for BOTH exponents
            e_lo = int(np.clip(e_lo, min_e, max_e))
            e_hi = int(np.clip(e_hi, min_e, max_e))

            cands.append((2.0 ** e_lo, e_lo, sign))
            cands.append((2.0 ** e_hi, e_hi, sign))

        # Pick best candidate by abs error in magnitude space
        best_mag, best_e, best_s = cands[0]
        best_err = abs(a - best_mag)
        for mag, e, s in cands[1:]:
            err = abs(a - mag)
            if err < best_err:
                best_err = err
                best_mag, best_e, best_s = mag, e, s

        if best_mag == 0.0 or best_s == 0:
            return 0.0, 0, 0

        # Final safety: exponent must be in range when non-zero
        if not (min_e <= best_e <= max_e):
            raise RuntimeError(
                f"PoT exponent out of range: {best_e}, expected [{min_e},{max_e}]"
            )

        return float(best_s * best_mag), int(best_e), int(best_s)

    def _quantize_apot(self, val):
        """
        Quantize slope to the nearest representable APoT value within exponent range,
        under DISTINCT-exponent constraint (each exponent used at most once).

        Representable magnitudes:
          {0} U { sum_{e in S} 2^e | S is a subset of [min_exp..max_exp] }

        Strategy:
          - Build candidates:
              1) 0
              2) PoT neighbors (single exponent) as candidates (range-safe)
              3) APoT floor: greedy largest <= |val| using distinct exps
              4) APoT ceil : minimal > |val| by adding smallest unused exps to floor set
          - Compare all candidates and choose closest to |val|.

        Return:
          (quantized_float, sorted_exponent_list, sign_int)

        Notes:
          - If 0 is chosen, exponent list is [] and sign is 0.
          - Exponent list is always sorted ascending.
        """
        if val == 0:
            return 0.0, [], 0

        sign = int(np.sign(val))
        a = abs(val)

        min_e = self._pot_min_exp
        max_e = self._pot_max_exp
        min_mag = 2.0 ** min_e
        max_mag = 2.0 ** max_e

        # Candidate list: (mag_value, exps_list)
        cands = [(0.0, [])]

        # ---- PoT neighbors as candidates (single exponent) ----
        # Range-safe:
        if a <= min_mag:
            cands.append((min_mag, [int(min_e)]))
        elif a >= max_mag:
            cands.append((max_mag, [int(max_e)]))
        else:
            e_lo = int(np.floor(np.log2(a)))
            e_hi = e_lo + 1

            # HARD clamp both
            e_lo = int(np.clip(e_lo, min_e, max_e))
            e_hi = int(np.clip(e_hi, min_e, max_e))

            cands.append((2.0 ** e_lo, [int(e_lo)]))
            cands.append((2.0 ** e_hi, [int(e_hi)]))

        # ---- APoT floor: greedy <= a, distinct exps ----
        def build_floor(a_mag):
            remaining = a_mag
            chosen = []
            v = 0.0
            for e in range(max_e, min_e - 1, -1):
                p = 2.0 ** e
                if remaining >= p:
                    v += p
                    remaining -= p
                    chosen.append(int(e))
            return v, sorted(chosen)

        # ---- APoT ceil: minimal > a by adding smallest unused exps ----
        def build_ceil(a_mag, floor_val, floor_exps):
            if floor_val > a_mag:
                return floor_val, floor_exps
            used = set(floor_exps)
            v = float(floor_val)
            exps = list(floor_exps)
            for e in range(min_e, max_e + 1):
                if e in used:
                    continue
                v += 2.0 ** e
                exps.append(int(e))
                used.add(e)
                if v > a_mag:
                    break
            return v, sorted(exps)

        v_lo, exps_lo = build_floor(a)
        cands.append((v_lo, exps_lo))

        v_hi, exps_hi = build_ceil(a, v_lo, exps_lo)
        if v_hi > a:
            cands.append((v_hi, exps_hi))

        # Choose the closest magnitude candidate
        best_v, best_exps = cands[0]
        best_err = abs(a - best_v)
        for v, exps in cands[1:]:
            err = abs(a - v)
            if err < best_err:
                best_err = err
                best_v, best_exps = v, exps

        if best_v == 0.0:
            return 0.0, [], 0

        # Safety: all exps must be within range
        for e in best_exps:
            if not (min_e <= int(e) <= max_e):
                raise RuntimeError(
                    f"APoT exponent out of range: {e}, expected [{min_e},{max_e}]"
                )

        return float(sign * best_v), list(best_exps), int(sign)

    # =====================================================================
    # Quantization cache construction (IMPORTANT: INT outputs stored here)
    # =====================================================================
    def _ensure_quantized(self, mode):
        """
        Build quantized caches for "PoT" or "APoT" once and store:
          - quantized slopes
          - quantized intercepts (continuous at left knot)
          - PoT exponent / APoT exponent list
          - sign array

        This guarantees:
          slopes("PoT") matches slopes("PoT-INT") + slopes("PoT-SIGN")
          slopes("APoT") matches slopes("APoT-INT") + slopes("APoT-SIGN")
        """
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
                qv, qe, qs = self._quantize_pot(v)

                # Extra safety: never allow exponent out of range for non-zero slopes
                if qs != 0 and not (self._pot_min_exp <= qe <= self._pot_max_exp):
                    raise RuntimeError(
                        f"PoT exponent out of range: {qe}, expected [{self._pot_min_exp},{self._pot_max_exp}]"
                    )

                mq[i] = qv
                exp[i] = qe
                sgn[i] = qs

            bq = np.empty_like(mq, dtype=float)
            for i in range(len(mq)):
                # Keep continuity at left knot i
                bq[i] = by[i] - mq[i] * bx[i]

            self._qcache["PoT"] = {"m": mq, "b": bq, "exp": exp, "sign": sgn}

        elif mode == "APoT":
            mq = np.empty_like(self._slopes, dtype=float)
            exps_obj = np.empty(len(self._slopes), dtype=object)
            sgn = np.empty_like(self._slopes, dtype=int)

            for i, v in enumerate(self._slopes):
                qv, exps_sorted, qs = self._quantize_apot(v)
                mq[i] = qv
                exps_obj[i] = list(exps_sorted)  # always a list, may be empty
                sgn[i] = qs

            bq = np.empty_like(mq, dtype=float)
            for i in range(len(mq)):
                bq[i] = by[i] - mq[i] * bx[i]

            self._qcache["APoT"] = {"m": mq, "b": bq, "exps": exps_obj, "sign": sgn}

    # =====================================================================
    # API: slopes / intercepts
    # =====================================================================
    def slopes(self, mode="original"):
        """
        Return slopes in different modes:
          - original: float slopes
          - PoT/APoT: float slopes
          - PoT-INT: exponent per segment (int)
          - APoT-INT: exponent list per segment (object array)
          - PoT-SIGN / APoT-SIGN: sign per segment (-1,0,+1)
        """
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

        # APoT family
        self._ensure_quantized("APoT")
        if mode == "APoT":
            return self._qcache["APoT"]["m"].copy()
        if mode == "APoT-INT":
            return self._qcache["APoT"]["exps"].copy()
        return self._qcache["APoT"]["sign"].copy()

    def intercepts(self, mode="original"):
        """
        Return intercepts in modes: "original", "PoT", "APoT"
        """
        mode = str(mode)
        if mode not in ("original", "PoT", "APoT"):
            raise ValueError('intercepts(mode) supports only: "original","PoT","APoT".')

        if mode == "original":
            return self._intercepts.copy()

        self._ensure_quantized(mode)
        return self._qcache[mode]["b"].copy()

    # =====================================================================
    # Interpolation helper (build nonlinear y on arbitrary xq)
    # =====================================================================
    def _interp_y(self, xq):
        """
        Interpolate the nonlinear function y(x) from sampled points (self.x, self.y).

        Uses np.interp for in-range values.
        For out-of-range values, uses linear extrapolation based on edge slopes.
        """
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

    # =====================================================================
    # Geometry helpers for greedy breakpoint selection
    # =====================================================================
    @staticmethod
    def _line_params(x0, y0, x1, y1):
        """
        Return line coefficients (A,B,C) for Ax + By + C = 0 and denom=sqrt(A^2+B^2).
        """
        A = y0 - y1
        B = x1 - x0
        C = x0 * y1 - x1 * y0
        denom = np.sqrt(A * A + B * B)
        return A, B, C, denom

    def _perp_dist(self, x_pts, y_pts, x0, y0, x1, y1):
        """
        Perpendicular distance from points (x_pts,y_pts) to the line through (x0,y0)-(x1,y1).
        """
        A, B, C, denom = self._line_params(x0, y0, x1, y1)
        if denom == 0:
            return np.zeros_like(x_pts, dtype=float)
        return np.abs(A * x_pts + B * y_pts + C) / denom

    # =====================================================================
    # Candidate computation
    # =====================================================================
    def _compute_top1_candidate(self, xl, xr):
        """
        For a segment [xl, xr], build the chord line connecting (xl, y(xl)) and (xr, y(xr)),
        then find the point in (xl, xr) with maximum perpendicular distance to that line.
        Return (dist_best, xb_int) where xb_int is the rounded breakpoint x coordinate.
        """
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

        # Must be strictly inside
        if not (xl < xb < xr):
            return None

        return (dist_best, xb)

    def _is_valid_breakpoint(self, xb, used_set):
        """
        Enforce minimum gap to all used breakpoints/endpoints.
        """
        for u in used_set:
            if abs(int(xb) - int(u)) <= self.min_gap:
                return False
        return True

    # =====================================================================
    # Greedy incremental breakpoint selection
    # =====================================================================
    def _fit_incremental(self):
        """
        Greedily add breakpoints until reaching target number, or no valid breakpoint exists.

        Policy:
          - For each current segment, compute its top-1 best breakpoint candidate (max distance).
          - Collect all segment top-1 candidates, pick the globally best valid candidate.
          - If chosen candidate is invalid (too close to existing breaks), try next best candidate
            from other segments (NOT from the same segment, as per your preference).
        """
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

            # Gather top-1 candidate from each segment
            per_seg = []
            for seg in segments:
                cand = self._candidate_cache.get(seg, None)
                if cand is None:
                    continue
                dist_val, xb = cand
                per_seg.append((dist_val, xb, seg))

            if not per_seg:
                break

            # Sort candidates by distance descending
            per_seg.sort(key=lambda t: -t[0])

            # Choose the first valid candidate among segments
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

            # Insert breakpoint
            self._breaks_x.append(int(xb))
            self._breaks_x = sorted(set(self._breaks_x))
            self._breaks_history.append(np.array(self._breaks_x, dtype=int))

            # Split segment
            xl, xr = seg_to_split
            left_seg = (xl, float(xb))
            right_seg = (float(xb), xr)

            segments.remove(seg_to_split)
            segments.append(left_seg)
            segments.append(right_seg)

            # Update candidate cache only for affected segments
            self._candidate_cache.pop(seg_to_split, None)
            self._candidate_cache[left_seg] = self._compute_top1_candidate(*left_seg)
            self._candidate_cache[right_seg] = self._compute_top1_candidate(*right_seg)

    # =====================================================================
    # Build final PWLF from breakpoints
    # =====================================================================
    def _rebuild_pwlf(self):
        """
        Build final knots_x/knots_y and compute slopes/intercepts for each segment.
        """
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

    # =====================================================================
    # Evaluation
    # =====================================================================
    def __call__(self, x_in, mode="original"):
        """
        Evaluate PWLF at x_in in specified mode.
        Mode: "original" | "PoT" | "APoT"
        """
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

    # =====================================================================
    # Plotting
    # =====================================================================
    def plot_final(self, mode="original", title=None):
        """
        Plot original nonlinear curve and the final PWLF in specified mode.
        """
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
        """
        Plot each greedy step (original PWLF only).
        """
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
        """
        Plot 4 panels:
          1) original nonlinear function
          2) fitted PWLF (original)
          3) PoT-approximated PWLF
          4) APoT-approximated PWLF
        """
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

    # =====================================================================
    # Debug: verify INT representation matches float slopes
    # =====================================================================
    def check_quant_int_consistency(self, verbose=True):
        """
        Check that PoT-INT/APoT-INT are consistent with PoT/APoT float slopes.

        For PoT:
          m_float == sign * 2^exp   (or 0 when sign=0)

        For APoT:
          m_float == sign * sum(2^e for e in exps_list)  (or 0 for empty list/sign=0)

        Prints max absolute reconstruction error.
        """
        # PoT check
        mp = self.slopes("PoT")
        ep = self.slopes("PoT-INT")
        sp = self.slopes("PoT-SIGN")
        mp_recon = sp * (2.0 ** ep)
        mp_recon = np.where(sp == 0, 0.0, mp_recon)
        pot_err = np.max(np.abs(mp - mp_recon))

        # APoT check
        ma = self.slopes("APoT")
        ea = self.slopes("APoT-INT")
        sa = self.slopes("APoT-SIGN")

        ma_recon = np.zeros_like(ma, dtype=float)
        for i in range(len(ma)):
            if sa[i] == 0:
                ma_recon[i] = 0.0
            else:
                exps = ea[i]
                if exps is None or len(exps) == 0:
                    ma_recon[i] = 0.0
                else:
                    ma_recon[i] = sa[i] * sum((2.0 ** int(e)) for e in exps)
        apot_err = np.max(np.abs(ma - ma_recon))

        if verbose:
            print("PoT INT consistency max abs error :", pot_err)
            print("APoT INT consistency max abs error:", apot_err)

        return pot_err, apot_err