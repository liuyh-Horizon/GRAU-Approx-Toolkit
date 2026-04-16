import numpy as np
import torch
import torch.nn as nn

from .grau_pwlf import GRAU_PWLF


class GRAU_KANLayer(nn.Module):
    """
    GRAU_KANLayer
    =============
    This module approximates each edge activation function in one KAN layer
    using an independent GRAU_PWLF object.

    Assumed KAN layer API:
      - kan_layer.in_dim
      - kan_layer.out_dim
      - kan_layer.get_subset([i], [j]) returns a sub-layer for one edge i -> j
      - sub_layer(x_edge) returns (y, _, _, _), where y has shape [B, 1]

    Main idea:
      For a KAN layer with in_dim = I and out_dim = O, we fit one PWLF per edge:
          phi_{i,j}(x)
      Then in forward:
          y_j = sum_i phi_{i,j}(x_i)

    Supported modes:
      - "PWLF"      : use original fitted PWLF
      - "PoT-PWLF"  : use PoT-quantized slopes
      - "APoT-PWLF" : use APoT-quantized slopes

    Initialization:
      GRAU_KANLayer(
          kan_layer,
          min_max_stats,
          device,
          segment_num=6,
          n_fit_points=1000,
          min_gap=1,
          min_improvement=0.0,
          pot_min_exp=-10,
          pot_max_exp=6
      )

    Notes:
      - Output is a normal torch.Tensor, not a QuantTensor.
      - min_max_stats is assumed to be a tuple/list: (xmin, xmax)
      - If your KAN layer uses extra bias/scale after edge summation,
        you may need to add that manually in forward().
    """

    def __init__(
        self,
        kan_layer,
        min_max_stats,
        device,
        segment_num=6,
        n_fit_points=1000,
        min_gap=1,
        min_improvement=0.0,
        pot_min_exp=-10,
        pot_max_exp=6,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Store original KAN layer
        # ------------------------------------------------------------
        self.kan_layer = kan_layer
        self.device = torch.device(device)

        # ------------------------------------------------------------
        # Basic dimensions
        # ------------------------------------------------------------
        self.in_dim = int(self._infer_in_dim())
        self.out_dim = int(self._infer_out_dim())

        # ------------------------------------------------------------
        # PWLF fitting configurations
        # ------------------------------------------------------------
        self.segment_num = int(segment_num)
        self.n_fit_points = int(n_fit_points)
        self.min_gap = int(min_gap)
        self.min_improvement = float(min_improvement)
        self.pot_min_exp = int(pot_min_exp)
        self.pot_max_exp = int(pot_max_exp)

        # ------------------------------------------------------------
        # Fitting x-range
        # ------------------------------------------------------------
        self.fit_xmin, self.fit_xmax = self._parse_range(min_max_stats)

        # ------------------------------------------------------------
        # Build one PWLF per edge (i -> j)
        # ------------------------------------------------------------
        self._fit_all_edges()

        # ------------------------------------------------------------
        # Build torch caches for fast forward
        # ------------------------------------------------------------
        self._build_torch_cache()

        self.to(self.device)

    # ============================================================
    # 1) Basic helpers
    # ============================================================
    def _infer_in_dim(self):
        if hasattr(self.kan_layer, "in_dim"):
            return int(self.kan_layer.in_dim)
        raise ValueError("kan_layer must provide attribute 'in_dim'.")

    def _infer_out_dim(self):
        if hasattr(self.kan_layer, "out_dim"):
            return int(self.kan_layer.out_dim)
        raise ValueError("kan_layer must provide attribute 'out_dim'.")

    def _parse_range(self, min_max_stats):
        """
        min_max_stats must be (xmin, xmax) for this KAN version.
        """
        if isinstance(min_max_stats, (tuple, list)) and len(min_max_stats) == 2:
            return float(min_max_stats[0]), float(min_max_stats[1])
        raise ValueError("min_max_stats must be (xmin, xmax).")

    # ============================================================
    # 2) Edge function evaluation helper
    # ============================================================
    @torch.no_grad()
    def _eval_single_edge_exact(self, i, j, x_1d):
        """
        Evaluate the exact KAN edge function phi_{i,j}(x) using the original KAN layer.

        Input:
          x_1d: torch.Tensor with shape [B] or [B,1]

        Return:
          y_1d: torch.Tensor with shape [B]
        """
        if x_1d.dim() == 1:
            x_1d = x_1d.unsqueeze(1)  # [B,1]

        sub_layer = self.kan_layer.get_subset([i], [j]).to(self.device)

        out = sub_layer(x_1d)

        # The user-provided code suggests:
        #   y, _, _, _ = sub_layer(x)
        if isinstance(out, (tuple, list)):
            y = out[0]
        else:
            y = out

        if y.dim() != 2 or y.size(1) != 1:
            raise ValueError(
                f"Expected edge output shape [B,1], but got {tuple(y.shape)} for edge {i}->{j}."
            )

        return y[:, 0]

    # ============================================================
    # 3) Fit one GRAU_PWLF per edge
    # ============================================================
    @torch.no_grad()
    def _fit_all_edges(self):
        """
        Fit one GRAU_PWLF object for each edge i -> j.
        """
        x = np.linspace(self.fit_xmin, self.fit_xmax, self.n_fit_points, dtype=np.float32)
        x_t = torch.tensor(x, device=self.device, dtype=torch.float32)  # [N]

        # 2D list:
        #   self.pwlf[i][j] = GRAU_PWLF for edge i -> j
        self.pwlf = [[None for _ in range(self.out_dim)] for _ in range(self.in_dim)]

        for i in range(self.in_dim):
            for j in range(self.out_dim):
                y_t = self._eval_single_edge_exact(i, j, x_t)  # [N]
                y = y_t.detach().cpu().numpy().astype(np.float32)

                model = GRAU_PWLF(
                    x=x,
                    y=y,
                    segment_num=self.segment_num,
                    require_integer=True,
                    sort_by_x=True,
                    min_improvement=self.min_improvement,
                    min_gap=self.min_gap,
                    pot_min_exp=self.pot_min_exp,
                    pot_max_exp=self.pot_max_exp,
                )
                self.pwlf[i][j] = model

    # ============================================================
    # 4) Build torch cache
    # ============================================================
    def _build_torch_cache(self):
        """
        Cache knots_x / slopes / intercepts for each edge and each mode.

        Structure:
          self._cache["PWLF"][i][j]      = (knots_x, slopes, intercepts)
          self._cache["PoT-PWLF"][i][j]  = ...
          self._cache["APoT-PWLF"][i][j] = ...
        """
        self._cache = {
            "PWLF": [[None for _ in range(self.out_dim)] for _ in range(self.in_dim)],
            "PoT-PWLF": [[None for _ in range(self.out_dim)] for _ in range(self.in_dim)],
            "APoT-PWLF": [[None for _ in range(self.out_dim)] for _ in range(self.in_dim)],
        }

        for i in range(self.in_dim):
            for j in range(self.out_dim):
                m = self.pwlf[i][j]

                knots_x = torch.tensor(m.knots_x, device=self.device, dtype=torch.float32)

                slopes = torch.tensor(m.slopes("original"), device=self.device, dtype=torch.float32)
                inters = torch.tensor(m.intercepts("original"), device=self.device, dtype=torch.float32)
                self._cache["PWLF"][i][j] = (knots_x, slopes, inters)

                slopes = torch.tensor(m.slopes("PoT"), device=self.device, dtype=torch.float32)
                inters = torch.tensor(m.intercepts("PoT"), device=self.device, dtype=torch.float32)
                self._cache["PoT-PWLF"][i][j] = (knots_x, slopes, inters)

                slopes = torch.tensor(m.slopes("APoT"), device=self.device, dtype=torch.float32)
                inters = torch.tensor(m.intercepts("APoT"), device=self.device, dtype=torch.float32)
                self._cache["APoT-PWLF"][i][j] = (knots_x, slopes, inters)

    # ============================================================
    # 5) Evaluate one edge PWLF on torch tensor
    # ============================================================
    def _eval_pwlf_edge_torch(self, x_1d, i, j, mode):
        """
        Evaluate one fitted edge PWLF on input x_1d.

        Input:
          x_1d: torch.Tensor [B]
        Return:
          y_1d: torch.Tensor [B]
        """
        if mode not in self._cache:
            raise ValueError('mode must be "PWLF", "PoT-PWLF", or "APoT-PWLF".')

        knots_x, slopes, inters = self._cache[mode][i][j]

        seg = torch.searchsorted(knots_x, x_1d, right=True) - 1
        seg = torch.clamp(seg, 0, slopes.numel() - 1)

        y_1d = slopes[seg] * x_1d + inters[seg]
        return y_1d

    # ============================================================
    # 6) Forward
    # ============================================================
    def forward(self, x, mode="PWLF"):
        """
        Forward of GRAU_KANLayer.

        Input:
          x: torch.Tensor with shape [B, in_dim]
        Output:
          y: torch.Tensor with shape [B, out_dim]

        Assumed KAN aggregation:
          y_j = sum_i phi_{i,j}(x_i)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch.Tensor.")

        x = x.to(self.device)

        if x.dim() != 2 or x.size(1) != self.in_dim:
            raise ValueError(f"Expected input shape [B,{self.in_dim}], but got {tuple(x.shape)}.")

        B = x.size(0)
        y = torch.zeros(B, self.out_dim, device=self.device, dtype=torch.float32)

        for i in range(self.in_dim):
            x_i = x[:, i]  # [B]
            for j in range(self.out_dim):
                y[:, j] += self._eval_pwlf_edge_torch(x_i, i, j, mode)

        return y

    # ============================================================
    # 7) Debug utilities
    # ============================================================
    def print_activation_pwlf_params(self, max_inputs=None, max_outputs=None, title_prefix="KANLayer"):
        """
        Print PWLF parameters of edges i -> j.

        max_inputs / max_outputs:
          - None: print all
          - int : print only a prefix range
        """
        I = self.in_dim if max_inputs is None else min(self.in_dim, int(max_inputs))
        O = self.out_dim if max_outputs is None else min(self.out_dim, int(max_outputs))

        print(f"[{title_prefix}] PWLF parameter dump")
        print(f"  in_dim={self.in_dim}, out_dim={self.out_dim}")
        print(f"  Fit range: [{self.fit_xmin}, {self.fit_xmax}] | n_fit_points={self.n_fit_points}")
        print(f"  PoT/APoT exp range: [{self.pot_min_exp}, {self.pot_max_exp}]")
        print("")

        for i in range(I):
            for j in range(O):
                m = self.pwlf[i][j]
                print(f"--- Edge {i}->{j} ---")
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

    def plot_all_act(self, layer_idx=0, max_inputs=None, max_outputs=None):
        """
        Plot compare figures for all selected KAN edges.

        Title format:
          L{layer_idx} edge i->j
        """
        I = self.in_dim if max_inputs is None else min(self.in_dim, int(max_inputs))
        O = self.out_dim if max_outputs is None else min(self.out_dim, int(max_outputs))

        for i in range(I):
            for j in range(O):
                self.pwlf[i][j].plot_compare(f"L{layer_idx} edge {i}->{j}")

    @torch.no_grad()
    def compare_with_exact(self, x, mode="PWLF"):
        """
        Compare the approximated GRAU_KANLayer output with the exact KAN edge-sum output.

        This is a debugging helper.

        Input:
          x: [B, in_dim]

        Return:
          y_exact, y_approx
        """
        x = x.to(self.device)

        if x.dim() != 2 or x.size(1) != self.in_dim:
            raise ValueError(f"Expected input shape [B,{self.in_dim}], but got {tuple(x.shape)}.")

        B = x.size(0)

        # exact output by summing original edge functions
        y_exact = torch.zeros(B, self.out_dim, device=self.device, dtype=torch.float32)
        for i in range(self.in_dim):
            x_i = x[:, i]
            for j in range(self.out_dim):
                y_exact[:, j] += self._eval_single_edge_exact(i, j, x_i)

        y_approx = self.forward(x, mode=mode)

        return y_exact, y_approx