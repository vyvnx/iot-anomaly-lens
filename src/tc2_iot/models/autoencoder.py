from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import AnomalyModel


def _net(d: int, hidden: list[int]) -> nn.Sequential:
    dims = [d, *hidden]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(dims[-1], d))  # linear output: standardized inputs can be negative
    return nn.Sequential(*layers)


class Autoencoder(AnomalyModel):
    name = "autoencoder"

    def __init__(self, params: dict, seed: int, n_features: int, device: str = "cpu"):
        hidden = list(params["hidden_dimensions"])
        bottleneck = min(hidden)
        if n_features <= bottleneck:
            raise ValueError(f"d={n_features} <= gargalo {bottleneck}: não há compressão; revise a arquitetura")
        self.params, self.seed, self.d, self.device = dict(params), seed, n_features, device
        torch.manual_seed(seed)
        self.net = _net(n_features, hidden).to(device)
        self.history: list[dict] = []
        self.best_epoch: int | None = None

    def fit(self, X_train, X_validation_fit=None):
        if X_validation_fit is None or len(X_validation_fit) == 0:
            raise ValueError("autoencoder exige validation_fit para early stopping")
        p = self.params
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        Xt = torch.as_tensor(X_train, dtype=torch.float32)
        Xv = torch.as_tensor(X_validation_fit, dtype=torch.float32).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        loss_fn = nn.MSELoss()
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, p["max_epochs"] + 1):
            t0 = time.perf_counter()
            self.net.train()
            perm = torch.randperm(len(Xt), generator=gen)  # shuffle only the training set
            total = 0.0
            for i in range(0, len(Xt), p["batch_size"]):
                xb = Xt[perm[i:i + p["batch_size"]]].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), xb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            val = float(self._mse(Xv).mean())
            improved = val < best - p["min_delta"]
            if improved:
                best, best_state, bad, self.best_epoch = val, copy.deepcopy(self.net.state_dict()), 0, epoch
            else:
                bad += 1
            self.history.append({"epoch": epoch, "train_mse": total / len(Xt), "validation_fit_mse": val,
                                 "improved": improved, "epoch_seconds": time.perf_counter() - t0})
            if bad >= p["patience"]:
                break
        self.net.load_state_dict(best_state)  # restore checkpoint with lowest validation loss
        return self

    @torch.no_grad()
    def _mse(self, X: torch.Tensor, batch: int = 65536) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(self.device)
            out.append(((self.net(xb) - xb) ** 2).mean(dim=1).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    def anomaly_score(self, X):
        return self._mse(torch.as_tensor(X, dtype=torch.float32))

    def save(self, path: Path):
        path = Path(path)
        w, meta = path / "state_dict.pt", path / "autoencoder.json"
        torch.save(self.net.state_dict(), w)
        meta.write_text(json.dumps({"params": self.params, "seed": self.seed, "d": self.d, "device": self.device,
                                    "history": self.history, "best_epoch": self.best_epoch}, indent=2))
        return [w, meta]

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        meta = json.loads((path / "autoencoder.json").read_text())
        m = cls(meta["params"], meta["seed"], meta["d"], meta["device"])
        m.net.load_state_dict(torch.load(path / "state_dict.pt", map_location=meta["device"], weights_only=True))
        m.history, m.best_epoch = meta["history"], meta["best_epoch"]
        return m

    def resolved_params(self):
        return {"architecture": f"{self.d} -> " + " -> ".join(map(str, self.params["hidden_dimensions"])) + f" -> {self.d}",
                "activation_hidden": "ReLU", "output": "linear", "loss": "MSE", "optimizer": "Adam",
                "early_stopping": "MSE em validation_fit; melhora se val < melhor - min_delta (absoluto)",
                "device": self.device, "seed": self.seed, "torch": torch.__version__, **self.params}

    def fit_info(self):
        return {"best_epoch": self.best_epoch, "epochs_run": len(self.history),
                "best_validation_fit_mse": min((h["validation_fit_mse"] for h in self.history), default=None)}
