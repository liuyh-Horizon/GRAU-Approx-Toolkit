import numpy as np
import torch
import torch.nn as nn

from brevitas.quant_tensor.int_quant_tensor import IntQuantTensor

from .grau_pwlf import GRAU_PWLF


class GRAU_ResConv2D(nn.Module):
    """
    GRAU_ResConv2D
    ===========
    This module fuses "BN + QuantAct" into a per-channel piecewise-linear function (PWLF)
    for a convolution layer output.

    Supported inference modes:
      - "PWLF"      : original fitted PWLF slopes
      - "PoT-PWLF"  : PoT-quantized slopes
      - "APoT-PWLF" : APoT-quantized slopes

    Forward pipeline:
      1) out_q  = conv_layer(x)                         # Brevitas QuantTensor
      2) out_i  = integer float tensor extracted from out_q._pre_round_int_value
      3) y_hat  = PWLF(out_i)                           # approximate BN + QuantAct
      4) y_int  = round(y_hat)
      5) y_clp  = clamp(y_int, out_min, out_max)
      6) return IntQuantTensor(value=y_clp, scale=1.0, zero_point=0.0, ...)

    Output clamp range:
      - inferred from act_layer(dummy).bit_width and act_layer(dummy).signed_t
      - if inference fails, use manual out_act_bitwidth & is_signed

    Fitting notes:
      - We "draw" BN+Act curve with BN running stats per channel:
            x_bn = gamma * (x - mean)/sqrt(var+eps) + beta
      - x is sampled in the measured MAC range (min_max_stats), shared across channels
      - We fit one GRAU_PWLF per output channel (C_out)
    """

    def __init__(
        self,
        conv_layer,
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
        # Manual fallback
        out_act_bitwidth=None,
        is_signed=None,
    ):
        super().__init__()

        # Store original layers
        self.conv = conv_layer
        self.bn = bn_layer
        self.act = act_layer

        self.device = torch.device(device)

        # PWLF settings
        self.segment_num = int(segment_num)
        self.n_fit_points = int(n_fit_points)
        self.min_gap = int(min_gap)
        self.min_improvement = float(min_improvement)
        self.pot_min_exp = int(pot_min_exp)
        self.pot_max_exp = int(pot_max_exp)

        # Infer output channels (C_out)
        self.output_channels = self._infer_output_channels()

        # MAC range for fitting
        self.fit_xmin, self.fit_xmax = self._parse_range(min_max_stats)

        # Infer output quant clamp range from act dummy output
        self.out_act_bitwidth = None
        self.is_signed = None
        self.out_min = None
        self.out_max = None
        self._range_source = "unknown"

        ok = self._infer_output_range_from_dummy_act_output()
        if not ok:
            self._set_output_range_from_manual(out_act_bitwidth, is_signed)

        # Fit per-channel PWLF
        self._fit_per_channel_pwlf()

        # Build torch cache for fast evaluation
        self._build_torch_cache()

        self.to(self.device)

    # ============================================================
    # 1) Utilities
    # ============================================================
    def _infer_output_channels(self):
        """
        Prefer BN.num_features, else conv.out_channels, else conv.weight.shape[0].
        """
        if hasattr(self.bn, "num_features"):
            return int(self.bn.num_features)
        if hasattr(self.conv, "out_channels"):
            return int(self.conv.out_channels)
        if hasattr(self.conv, "weight") and self.conv.weight is not None:
            return int(self.conv.weight.shape[0])
        raise ValueError("Cannot infer output_channels from bn_layer or conv_layer.")

    def _parse_range(self, min_max_stats):
        """
        min_max_stats must be (xmin, xmax) for your final version.
        """
        if isinstance(min_max_stats, (tuple, list)) and len(min_max_stats) == 2:
            return float(min_max_stats[0]), float(min_max_stats[1])
        raise ValueError("min_max_stats must be (xmin, xmax).")

    @staticmethod
    def _to_python_scalar(v):
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
    # 2) QuantTensor -> integer float tensor
    # ============================================================
    @staticmethod
    def _quant_to_int_tensor(q):
        """
        Your final Linear version uses:
          - if hasattr(q, "_pre_round_int_value"):
              return round(q._pre_round_int_value).float()
          else: raise

        We keep the same policy here.
        """
        if hasattr(q, "_pre_round_int_value"):
            return torch.round(q._pre_round_int_value).type(torch.float32)
        raise TypeError("Cannot convert QuantTensor-like object to integer tensor: missing _pre_round_int_value.")

    # ============================================================
    # 3) Output range inference (dummy act output only)
    # ============================================================
    def _infer_output_range_from_dummy_act_output(self):
        """
        Infer clamp range from:
          y = act_layer(dummy)
          bw = y.bit_width
          sgn = y.signed_t
        """
        try:
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
    # 4) Fitting PWLF per channel
    # ============================================================
    @torch.no_grad()
    def _act_pre_round_value(self, x_bn):
        """
        Apply QuantAct and return a tensor for fitting.
        Prefer _pre_round_int_value.
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
        Draw BN+Act curve and fit one GRAU_PWLF per output channel.
        """
        # Sample x in MAC range
        x = np.linspace(self.fit_xmin, self.fit_xmax, self.n_fit_points, dtype=np.float32)
        x_t = torch.tensor(x, device=self.device, dtype=torch.float32).view(-1, 1)  # [N,1]

        # BN params (Conv must use BatchNorm2d, but running stats are still per-channel)
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

        # Expand x to [N,C] and apply BN formula
        x_mat = x_t.repeat(1, C)  # [N,C]
        x_norm = (x_mat - mean) / torch.sqrt(var + eps)
        x_bn = gamma * x_norm + beta

        # Apply QuantAct and extract pre-round int curve
        y_pre = self._act_pre_round_value(x_bn)          # [N,C]
        y_np = y_pre.detach().cpu().numpy().astype(np.float32)

        # Fit PWLF per channel
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
    # 5) Torch cache & evaluation
    # ============================================================
    def _build_torch_cache(self):
        """
        Cache (knots_x, slopes, intercepts) for each channel and each mode.
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
        Evaluate PWLF on x.

        x shape for Conv2d: [B, C, H, W]
        We evaluate channel-by-channel:
          - flatten spatial dims to 1D
          - compute segment index by searchsorted(knots_x, ...)
          - y = m*xc + b
          - reshape back
        """
        if mode not in self._cache:
            raise ValueError('mode must be "PWLF", "PoT-PWLF", or "APoT-PWLF".')

        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}.")

        B, C, H, W = x.shape
        if C != self.output_channels:
            raise ValueError(f"Expected C={self.output_channels}, got C={C}.")

        y = torch.empty_like(x)

        for c in range(C):
            knots_x, slopes, inters = self._cache[mode][c]

            xc = x[:, c, :, :].reshape(-1)  # [B*H*W]

            seg = torch.searchsorted(knots_x, xc, right=True) - 1
            seg = torch.clamp(seg, 0, slopes.numel() - 1)

            yc = slopes[seg] * xc + inters[seg]
            y[:, c, :, :] = yc.reshape(B, H, W)

        return y

    # ============================================================
    # 6) Output quantization
    # ============================================================
    def _round_and_clamp(self, y):
        y = torch.round(y)
        y = torch.clamp(y, self.out_min, self.out_max)
        return y

    # ============================================================
    # 7) Forward
    # ============================================================
    def forward(self, x, identity, mode="PWLF"):
        """
        x: Brevitas QuantTensor input to conv
        mode: "PWLF" | "PoT-PWLF" | "APoT-PWLF"
        return: IntQuantTensor with integer value tensor
        """
        out_q = self.conv(x)

        # Convert QuantTensor -> integer float tensor [B,C,H,W]
        out_i = self._quant_to_int_tensor(out_q).to(self.device) + self._quant_to_int_tensor(identity).to(self.device)

        if out_i.dim() != 4 or out_i.size(1) != self.output_channels:
            raise ValueError(f"Expected [B,{self.output_channels},H,W], got {tuple(out_i.shape)}.")

        y_hat = self._eval_pwlf_torch(out_i, mode)
        y_out = self._round_and_clamp(y_hat)

        # Wrap back into IntQuantTensor (scale=1, zero_point=0 like your Linear version)
        y_q = IntQuantTensor(
            value=y_out,
            scale=1.0,
            zero_point=0.0,
            bit_width=self.out_act_bitwidth,
            signed=self.is_signed,
            training=False
        ).to(self.device)

        return y_q

    # ============================================================
    # 8) Debug utilities
    # ============================================================
    def print_activation_pwlf_params(self, title_prefix="Activation", max_channels=None):
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
        Plot compare figures for each channel.
        Warning: for large C_out, this will generate many figures.
        """
        C = self.output_channels
        if max_channels is not None:
            C = min(C, int(max_channels))

        for c in range(C):
            self.pwlf[c].plot_compare(f"{title_prefix} (ch={c})")