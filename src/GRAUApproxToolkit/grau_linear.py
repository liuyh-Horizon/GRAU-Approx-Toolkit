import numpy as np
import torch
import torch.nn as nn

from brevitas.quant_tensor.int_quant_tensor import IntQuantTensor

from .grau_pwlf import GRAU_PWLF


class GRAU_Linear(nn.Module):
    """
    GRAU_Linear
    ===========
    This module fuses "BN + QuantAct" into a piecewise-linear function (PWLF) per output channel.
    It supports 3 inference modes:
      - "PWLF"      : original fitted PWLF slopes
      - "PoT-PWLF"  : PoT-quantized slopes derived from the final PWLF
      - "APoT-PWLF" : APoT-quantized slopes derived from the final PWLF

    Overall pipeline (what happens in forward):
      1) out_q  = linear_layer(x)                           # Brevitas QuantTensor
      2) out_i  = out_q.int(float_datatype=True)            # IMPORTANT: integer-valued float Tensor
      3) y_hat  = PWLF(out_i)                               # approximate BN + QuantAct
      4) y_int  = round(y_hat)                              # integer-valued float Tensor
      5) y_clp  = clamp(y_int, out_min, out_max)            # enforce output integer range

    Why we need clamp:
      - We want the output to match the target activation integer range, such as:
          * signed 8-bit  : [-128, 127]
          * unsigned 8-bit: [0, 255]
      - The range is inferred from QuantAct output metadata (bit_width & signed_t).
      - If inference fails, you can provide (out_act_bitwidth, is_signed) manually.

    Initialization:
      GRAU_Linear(
        linear_layer, bn_layer, act_layer,
        min_max_stats, device,
        segment_num=6, n_fit_points=1000,
        min_gap=1, min_improvement=0.0,
        pot_min_exp=-10, pot_max_exp=6,
        out_act_bitwidth=None, is_signed=None
      )

    Notes about fitting:
      - We "draw" the nonlinear curve using BN running statistics:
            x_bn = gamma * (x - mean) / sqrt(var + eps) + beta
        and then apply QuantAct to get the *pre-round integer* values (if available).
      - This is consistent with your previous code style and avoids QuantTensor->BN issues.
    """

    def __init__(
        self,
        linear_layer,
        bn_layer,
        act_layer,
        min_max_stats,
        device,
        segment_num=6,
        n_fit_points=1000,
        min_gap=1,
        min_improvement=0.0,
        pot_min_exp=-10,
        pot_max_exp=6,
        # Manual fallback if (bit_width, signed_t) cannot be inferred from dummy act output
        out_act_bitwidth=None,
        is_signed=None,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Store original layers
        # ------------------------------------------------------------
        # Pass the trained model's real layers here.
        # This class will not build or modify these layers.
        self.linear = linear_layer
        self.bn = bn_layer
        self.act = act_layer

        # Where we run fitting & forward
        self.device = torch.device(device)

        # ------------------------------------------------------------
        # PWLF fitting configurations
        # ------------------------------------------------------------
        self.segment_num = int(segment_num)           # desired number of segments per channel
        self.n_fit_points = int(n_fit_points)         # number of sampled points for curve fitting
        self.min_gap = int(min_gap)                   # minimal distance between integer breakpoints
        self.min_improvement = float(min_improvement) # optional threshold for stopping early
        self.pot_min_exp = int(pot_min_exp)           # PoT/APoT exponent range
        self.pot_max_exp = int(pot_max_exp)

        # ------------------------------------------------------------
        # Infer channel count (C)
        # ------------------------------------------------------------
        self.output_channels = self._infer_output_channels()

        # ------------------------------------------------------------
        # Range of MAC output used to draw BN+Act curve
        # ------------------------------------------------------------
        self.fit_xmin, self.fit_xmax = self._parse_range(min_max_stats)

        # ------------------------------------------------------------
        # Infer output clamp range from QuantAct dummy output (bit_width & signed_t)
        # ------------------------------------------------------------
        # If inference fails, fall back to manual (out_act_bitwidth, is_signed).
        self.out_act_bitwidth = None
        self.is_signed = None
        self.out_min = None
        self.out_max = None
        self._range_source = "unknown"

        ok = self._infer_output_range_from_dummy_act_output()
        if not ok:
            self._set_output_range_from_manual(out_act_bitwidth, is_signed)

        # ------------------------------------------------------------
        # Fit per-channel PWLF models
        # ------------------------------------------------------------
        self._fit_per_channel_pwlf()

        # ------------------------------------------------------------
        # Build fast torch cache for forward evaluation
        # ------------------------------------------------------------
        self._build_torch_cache()

        # Move module to device (also moves BN/Linear/Act if they are registered modules)
        self.to(self.device)

    # ============================================================
    # 1) Small utilities: channel inference & range parsing
    # ============================================================
    def _infer_output_channels(self):
        """
        Infer output channels C.
        We prefer bn.num_features because it must match the activation dimension.
        """
        if hasattr(self.bn, "num_features"):
            return int(self.bn.num_features)
        if hasattr(self.linear, "out_features"):
            return int(self.linear.out_features)
        if hasattr(self.linear, "weight") and self.linear.weight is not None:
            return int(self.linear.weight.shape[0])
        raise ValueError("Cannot infer output_channels from bn_layer or linear_layer.")

    def _parse_range(self, min_max_stats):
        """
        min_max_stats can be:
          - (xmin, xmax)
        """
        if isinstance(min_max_stats, (tuple, list)) and len(min_max_stats) == 2:
            return float(min_max_stats[0]), float(min_max_stats[1])

        raise ValueError("min_max_stats must be (xmin,xmax).")

    @staticmethod
    def _to_python_scalar(v):
        """
        Convert tensor/np scalar -> python scalar.
        Return None if cannot convert.
        """
        try:
            if v is None:
                return None
            if isinstance(v, (int, float, bool, np.bool_)):
                return v
            if isinstance(v, np.generic):
                return v.item()
            if torch.is_tensor(v) and v.numel() == 1:
                return v.detach().cpu().item()
            return None
        except Exception:
            return None

    # ============================================================
    # 2) QuantTensor conversion (IMPORTANT for Brevitas)
    # ============================================================
    @staticmethod
    def _quant_to_int_tensor(q):
        """
        Convert Brevitas QuantTensor to integer-valued float torch.Tensor.

        In many Brevitas pipelines:
          QuantLinear returns QuantTensor
          -> you MUST call .int(float_datatype=True) before feeding to other ops
             (you explicitly mentioned this requirement).

        Priority:
          - if has .int():   use q.int(float_datatype=True)
        """
        
        # if hasattr(q, "int"):
        #     return q.int(float_datatype=True)

        if  hasattr(q, "_pre_round_int_value"):
            return torch.round(q._pre_round_int_value).type(torch.float32)
            
        raise TypeError("Cannot convert QuantTensor-like object to torch.Tensor with integer value. There is no .int()")

    # ============================================================
    # 3) Output range inference (only dummy output bit_width + signed_t)
    # ============================================================
    def _infer_output_range_from_dummy_act_output(self):
        """
        Try to infer output clamp range ONLY from:
          y = act_layer(dummy)
        and reading:
          - y.bit_width
          - y.signed_t

        If either is missing, return False.

        Why this is robust for your case:
          - You confirmed QuantAct output has bit_width and signed_t.
          - You confirmed there is NO min_val/max_val to read.
        """
        try:
            # dummy shape must match activation channels: [B, C]
            dummy = torch.zeros(1, self.output_channels, device=self.device, dtype=torch.float32)
            y = self.act(dummy)

            bw = self._to_python_scalar(getattr(y, "bit_width", None))
            sgn = self._to_python_scalar(getattr(y, "signed_t", None))

            if bw is None or sgn is None:
                return False

            bw = int(bw)
            sgn = bool(sgn)

            self.out_act_bitwidth = bw
            self.is_signed = sgn

            # integer clamp range
            if self.is_signed:
                self.out_min = -(2 ** (bw - 1))
                self.out_max = (2 ** (bw - 1)) - 1
            else:
                self.out_min = 0
                self.out_max = (2 ** bw) - 1

            self._range_source = "dummy_act_output(bit_width+signed_t)"
            return True

        except Exception:
            return False

    def _set_output_range_from_manual(self, out_act_bitwidth, is_signed):
        """
        Manual fallback if dummy inference fails.
        """
        if out_act_bitwidth is None or is_signed is None:
            raise ValueError(
                "Cannot infer output clamp range from act_layer dummy output. "
                "Please provide out_act_bitwidth and is_signed manually."
            )

        bw = int(out_act_bitwidth)
        sgn = bool(is_signed)

        self.out_act_bitwidth = bw
        self.is_signed = sgn

        if sgn:
            self.out_min = -(2 ** (bw - 1))
            self.out_max = (2 ** (bw - 1)) - 1
        else:
            self.out_min = 0
            self.out_max = (2 ** bw) - 1

        self._range_source = "manual(bitwidth+signed)"

    # ============================================================
    # 4) Fitting PWLF: BN running stats + QuantAct pre-round values
    # ============================================================
    @torch.no_grad()
    def _act_pre_round_value(self, x_bn):
        """
        Apply QuantAct and return a tensor suitable for fitting.

        Prefer:
          - QuantTensor._pre_round_int_value  (your old code used this)
        Else fallback:
          - QuantTensor.value
          - plain torch.Tensor
        """
        y = self.act(x_bn)

        if hasattr(y, "_pre_round_int_value"):
            return y._pre_round_int_value
        if hasattr(y, "value"):
            return y.value
        if torch.is_tensor(y):
            return y

        raise TypeError("Unsupported activation output type.")

    @torch.no_grad()
    def _fit_per_channel_pwlf(self):
        """
        Draw curve and fit PWLF per channel.

        Steps:
          1) x: sample MAC range uniformly (N points)
          2) expand x to [N,C]
          3) apply BN formula using running stats
          4) apply QuantAct to get pre-round int values y_pre
          5) for each channel c: fit PWLF model from (x, y_pre[:,c])
        """
        # 1) sample x
        x = np.linspace(self.fit_xmin, self.fit_xmax, self.n_fit_points, dtype=np.float32)
        x_t = torch.tensor(x, device=self.device, dtype=torch.float32).view(-1, 1)  # [N,1]

        # print("Dummy Input: ", x_t)
        # print("Dummy Input shape", x_t.shape)

        # 2) BN parameters
        bn = self.bn
        if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            raise TypeError("bn_layer must be a torch.nn.BatchNorm* instance.")

        eps = float(bn.eps)
        gamma = bn.weight.detach().to(self.device).view(1, -1)        # [1,C]
        beta  = bn.bias.detach().to(self.device).view(1, -1)          # [1,C]
        mean  = bn.running_mean.detach().to(self.device).view(1, -1)  # [1,C]
        var   = bn.running_var.detach().to(self.device).view(1, -1)   # [1,C]

        C = self.output_channels
        if gamma.numel() != C:
            raise ValueError(f"BN channel mismatch: bn has {gamma.numel()} channels, inferred C={C}.")

        # 3) expand x
        x_mat = x_t.repeat(1, C)  # [N,C]

        # 4) BN formula
        x_norm = (x_mat - mean) / torch.sqrt(var + eps)
        x_bn = gamma * x_norm + beta

        # print("BN Result", x_bn)
        # print("BN Result shape:", x_bn.shape)

        # 5) QuantAct outputs (pre-round integer values)
        y_pre = self._act_pre_round_value(x_bn)  # [N,C]

        # print("Act Result", x_bn)
        # print("Act Result shape:", x_bn.shape)
        
        y_np = y_pre.detach().cpu().numpy().astype(np.float32)

        # 6) fit PWLF per channel
        self.pwlf = []
        for c in range(C):
            model = GRAU_PWLF(
                x=x,
                y=y_np[:, c],
                segment_num=self.segment_num,
                require_integer=True,
                sort_by_x=True,
                min_improvement=self.min_improvement,
                min_gap=self.min_gap,
                pot_min_exp=self.pot_min_exp,
                pot_max_exp=self.pot_max_exp,
            )
            self.pwlf.append(model)

    # ============================================================
    # 5) Build torch cache for fast forward evaluation
    # ============================================================
    def _build_torch_cache(self):
        """
        Cache (knots_x, slopes, intercepts) for each channel and each mode.
        This avoids numpy during forward and accelerates inference.
        """
        self._cache = {"PWLF": [], "PoT-PWLF": [], "APoT-PWLF": []}

        for c in range(self.output_channels):
            m = self.pwlf[c]
            knots_x = torch.tensor(m.knots_x, device=self.device, dtype=torch.float32)

            slopes = torch.tensor(m.slopes("original"), device=self.device, dtype=torch.float32)
            inters = torch.tensor(m.intercepts("original"), device=self.device, dtype=torch.float32)
            self._cache["PWLF"].append((knots_x, slopes, inters))

            slopes = torch.tensor(m.slopes("PoT"), device=self.device, dtype=torch.float32)
            inters = torch.tensor(m.intercepts("PoT"), device=self.device, dtype=torch.float32)
            self._cache["PoT-PWLF"].append((knots_x, slopes, inters))

            slopes = torch.tensor(m.slopes("APoT"), device=self.device, dtype=torch.float32)
            inters = torch.tensor(m.intercepts("APoT"), device=self.device, dtype=torch.float32)
            self._cache["APoT-PWLF"].append((knots_x, slopes, inters))

    def _eval_pwlf_torch(self, x, mode):
        """
        Evaluate the cached piecewise linear model on x (shape [B,C]).
        """
        if mode not in self._cache:
            raise ValueError('mode must be "PWLF", "PoT-PWLF", or "APoT-PWLF".')

        B, C = x.shape
        if C != self.output_channels:
            raise ValueError(f"Expected C={self.output_channels}, but got C={C}.")

        y = torch.empty_like(x)

        for c in range(C):
            knots_x, slopes, inters = self._cache[mode][c]
            xc = x[:, c]

            # Determine which segment each xc belongs to:
            # seg = index of last knot <= xc
            seg = torch.searchsorted(knots_x, xc, right=True) - 1
            seg = torch.clamp(seg, 0, slopes.numel() - 1)

            y[:, c] = slopes[seg] * xc + inters[seg]

        return y

    # ============================================================
    # 6) Output quantization: round + clamp
    # ============================================================
    def _round_and_clamp(self, y):
        """
        After PWLF evaluation, output must be integer and within [out_min, out_max].
        """
        y = torch.round(y)
        y = torch.clamp(y, self.out_min, self.out_max)
        return y

    # ============================================================
    # 7) Forward
    # ============================================================
    def forward(self, x, mode="PWLF"):
        """
        x: QuantTensor or torch.Tensor, usually shape [B, in_features]
        mode: "PWLF" | "PoT-PWLF" | "APoT-PWLF"

        Steps:
          1) out_q  = linear(x)
          2) out_i  = out_q.int(float_datatype=True)
          3) y_hat  = pwlf(out_i)
          4) y_int  = round + clamp
        """
        out_q = self.linear(x)

        # print("PWLF IntQuantTesnor:", out_q)

        # Critical conversion to integer float tensor:
        out_i = self._quant_to_int_tensor(out_q).to(self.device)

        # print("PWLF IntQuantTesnor of rounding:", out_i)

        if out_i.dim() != 2 or out_i.size(1) != self.output_channels:
            raise ValueError(f"Expected [B,{self.output_channels}], got {tuple(out_i.shape)}.")

        y_hat = self._eval_pwlf_torch(out_i, mode)
        y_out = self._round_and_clamp(y_hat)

        y_q_tensor = IntQuantTensor(value = y_out, 
                             scale=1.0, 
                             zero_point=0.0, 
                             bit_width=self.out_act_bitwidth, 
                             signed=self.is_signed, 
                             training=False).to(self.device)
        return y_q_tensor

    # ============================================================
    # 8) Debug utilities
    # ============================================================
    def print_activation_pwlf_params(self, title_prefix="Activation", max_channels=None):
        """
        Print per-channel breakpoints/slopes and the clamp range.
        """
        C = self.output_channels
        if max_channels is not None:
            C = min(C, int(max_channels))

        print(f"[{title_prefix}] PWLF parameter dump: channels={C}, target_segments={self.segment_num}")
        print(f"  Fit range: [{self.fit_xmin}, {self.fit_xmax}] | n_fit_points={self.n_fit_points}")
        print(f"  PoT/APoT exp range: [{self.pot_min_exp}, {self.pot_max_exp}]")
        print(f"  Output clamp range: [{self.out_min}, {self.out_max}] | source={self._range_source}")
        print(f"  (bit_width={self.out_act_bitwidth}, signed={self.is_signed})")
        print("")

        for c in range(C):
            m = self.pwlf[c]
            print(f"--- Channel {c} ---")
            print(f"segment_num: {m.segment_num}")
            print(f"breakpoints (internal): {m.breakpoints}")
            print(f"knots_x (with endpoints): {m.knots_x}")
            print("")
            print("slopes:")
            print(f"  PWLF      : {m.slopes('original')}")
            print(f"  PoT-PWLF  : {m.slopes('PoT')}")
            print(f"  APoT-PWLF : {m.slopes('APoT')}")
            print("")
            print("PoT representation:")
            print(f"  PoT-INT   : {m.slopes('PoT-INT')}")
            print(f"  PoT-SIGN  : {m.slopes('PoT-SIGN')}")
            print("")
            print("APoT representation:")
            print(f"  APoT-INT  : {m.slopes('APoT-INT')}")
            print(f"  APoT-SIGN : {m.slopes('APoT-SIGN')}")
            print("")

    def plot_all_act(self, title_prefix="Activation", max_channels=None):
        """
        Plot (Original / PWLF / PoT-PWLF / APoT-PWLF) for each channel via plot_compare().

        Warning:
          - If C is large (e.g., 256), this will generate many figures.
          - Use max_channels to limit plotting.
        """
        C = self.output_channels
        if max_channels is not None:
            C = min(C, int(max_channels))

        for c in range(C):
            self.pwlf[c].plot_compare(f"{title_prefix} (ch={c})")