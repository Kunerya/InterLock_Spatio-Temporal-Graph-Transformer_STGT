# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridTripletSimilarityLoss(nn.Module):
    """
    三元组 + (可选)成对回归 的混合损失，支持“双空间”：
      - Triplet 分支：triplet_space ∈ {"euclid", "cosine", "exp"}
          * euclid ：距离域，hinge(d_ap - d_an + m)
          * cosine ：相似度域，hinge(m - (s_ap - s_an))，s∈[-1,1]
          * exp    ：相似度域，s = exp(-α·d) ∈ (0,1]，同样用 hinge(m - Δs)
      - Pair 回归分支：pair_space ∈ {"exp", "cosine"}
          * exp    ：s = exp(-α·d)，直接回归到人工分数[0,1]
          * cosine ：余弦相似度先线性映射到[0,1]再回归
    """
    def __init__(self,
                 alpha: float = 1.0,            # 仅 exp 分支有效
                 margin: float = 0.1,           # 三元组 margin（欧氏为距离域；cos/exp 为相似度域）
                 lambda_pair: float = 0.3,
                 pair_loss: str = "mse",        # "mse" | "huber"
                 triplet_space: str = "euclid", # "euclid" | "cosine" | "exp"
                 pair_space: str = "exp",       # "exp" | "cosine"
                 eps: float = 1e-8):
        super().__init__()
        assert triplet_space in ("euclid", "cosine", "exp")
        assert pair_space in ("cosine", "exp")
        assert pair_loss in ("mse", "huber")
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.lambda_pair = float(lambda_pair)
        self.pair_loss = pair_loss
        self.triplet_space = triplet_space
        self.pair_space = pair_space
        self.eps = float(eps)

    # --- 距离 / 相似度 ---
    @staticmethod
    def _dist(a, b):
        return torch.norm(a - b, p=2, dim=-1)                    # [B]

    def _sim_exp(self, a, b):
        d = self._dist(a, b)
        x = torch.clamp(-self.alpha * d, min=-50.0, max=0.0)     # 防下溢/上溢
        return torch.exp(x)                                      # (0,1]

    def _sim_cos(self, a, b):
        a_n = F.normalize(a, p=2, dim=-1, eps=self.eps)
        b_n = F.normalize(b, p=2, dim=-1, eps=self.eps)
        return (a_n * b_n).sum(dim=-1)                           # [-1,1]

    def _sim(self, a, b, space):
        if space == "cosine":
            return self._sim_cos(a, b)
        elif space == "exp":
            return self._sim_exp(a, b)
        else:
            raise ValueError(f"_sim() only for cosine/exp, got {space}")

    def _pair_reg(self, pred, target):
        pred = pred.view(-1)
        target = torch.atleast_1d(target).to(pred.device, dtype=pred.dtype).view(-1)
        if self.pair_loss == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none")
        else:
            return F.mse_loss(pred, target, reduction="none")

    def forward(self, a, p, n, y_ap=None, y_an=None, weight=None):
        # 形状对齐
        if a.dim()==1: a=a.unsqueeze(0)
        if p.dim()==1: p=p.unsqueeze(0)
        if n.dim()==1: n=n.unsqueeze(0)
        B = a.size(0)

        # --- Triplet 分支 ---
        if self.triplet_space == "euclid":
            d_ap = self._dist(a, p)
            d_an = self._dist(a, n)
            tri = F.relu(d_ap - d_an + self.margin)              # [B]
        elif self.triplet_space == "cosine":
            s_ap = self._sim_cos(a, p)
            s_an = self._sim_cos(a, n)
            tri = F.relu(self.margin - (s_ap - s_an))            # [B]
        elif self.triplet_space == "exp":
            s_ap = self._sim_exp(a, p)
            s_an = self._sim_exp(a, n)
            tri = F.relu(self.margin - (s_ap - s_an))            # [B]
        else:
            raise ValueError(f"Unknown triplet_space: {self.triplet_space}")

        # 权重
        if weight is None:
            w = torch.ones(B, device=a.device, dtype=a.dtype)
        else:
            w = torch.atleast_1d(weight).to(a.device, dtype=a.dtype).view(-1)
            if w.numel() == 1:
                w = w.expand(B)

        tri_loss = (tri * w).mean()

        # --- Pair 回归分支 ---
        pair_terms = []

        def _pair_term(s_pred, space, y):
            if y is None:
                return None
            if space == "cosine":
                # 余弦 [-1,1] -> [0,1]
                s_pred = (s_pred + 1.0) * 0.5
            return self._pair_reg(s_pred, y)                     # [B]

        if y_ap is not None:
            if self.pair_space == "exp":
                s_ap_pair = self._sim_exp(a, p)
            else:
                s_ap_pair = self._sim_cos(a, p)
            t = _pair_term(s_ap_pair, self.pair_space, y_ap)
            if t is not None:
                pair_terms.append(t)

        if y_an is not None:
            if self.pair_space == "exp":
                s_an_pair = self._sim_exp(a, n)
            else:
                s_an_pair = self._sim_cos(a, n)
            t = _pair_term(s_an_pair, self.pair_space, y_an)
            if t is not None:
                pair_terms.append(t)

        if pair_terms:
            pair_loss = torch.stack(pair_terms, 0).mean()
        else:
            pair_loss = torch.zeros((), device=a.device, dtype=a.dtype)

        return tri_loss + self.lambda_pair * pair_loss
