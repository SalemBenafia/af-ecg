"""Optional deep-learning model for a Colab GPU: a small 1-D ResNet on the raw ECG.
    python -m afdetect.cnn --config configs/cnn.yaml          # train; keeps the best epoch on VAL
    python -m afdetect.cnn --config configs/cnn.yaml --test   # ONCE at the end: best model on TEST
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from .data import LABELS, read_wfdb
from .features import to_fs
from .metrics import summarize

WIN = 9000  # 30 s at 300 Hz


def load_signals(root, records):
    out = []
    for r in records:
        x, fs = read_wfdb(Path(root) / r)
        x = to_fs(x, fs)
        out.append(((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32))
    return out
class ECGSet(Dataset):
    """Random 30-s crop + random polarity flip when training; centre crop when evaluating."""

    def __init__(self, signals, labels, train):
        self.x, self.y, self.train = signals, labels, train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if len(x) >= WIN:
            s = int(torch.randint(0, len(x) - WIN + 1, (1,))) if self.train else (len(x) - WIN) // 2
            x = x[s:s + WIN]
        else:
            x = np.pad(x, (0, WIN - len(x)))
        if self.train and torch.rand(1).item() < 0.5:     # some devices are held upside down
            x = -x
        return torch.from_numpy(np.ascontiguousarray(x))[None], int(self.y[i])


class Block(nn.Module):
    def __init__(self, cin, cout, stride, p=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 15, stride, 7, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 15, 1, 7, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p)
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        h = self.drop(torch.relu(self.b1(self.c1(x))))
        return torch.relu(self.b2(self.c2(h)) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, n_classes=4, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, widths[0], 15, 2, 7, bias=False), nn.BatchNorm1d(widths[0]),
                                  nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        layers, cin = [], widths[0]
        for w in widths:
            layers += [Block(cin, w, 2), Block(w, w, 1)]
            cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    return np.concatenate([torch.softmax(model(xb.to(device)), 1).cpu().numpy() for xb, _ in loader])


def subset(cfg, splits, name):
    d = splits[splits["split"] == name].reset_index(drop=True)
    return load_signals(cfg["data_root"], d["record"]), d["label"].map(LABELS.index).to_numpy(), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    splits = pd.read_csv(cfg["split_file"])
    out_dir, model_path = Path(cfg["out_dir"]), Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bs, nw = cfg["batch_size"], cfg["num_workers"]
    model = ResNet1D().to(device)

    if not a.test:
        tr_x, tr_y, _ = subset(cfg, splits, "train")
        va_x, va_y, va = subset(cfg, splits, "val")
        counts = np.bincount(tr_y, minlength=4)
        weights = torch.tensor(len(tr_y) / (4 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)                  # classes are imbalanced
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        tl = DataLoader(ECGSet(tr_x, tr_y, True), batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
        vl = DataLoader(ECGSet(va_x, va_y, False), batch_size=bs, num_workers=nw)
        best, history = -1.0, []
        for ep in range(cfg["epochs"]):
            model.train()
            total = 0.0
            for xb, yb in tl:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    loss = loss_fn(model(xb.to(device)), yb.to(device))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(yb)
            sched.step()
            p = predict_proba(model, vl, device)
            s = summarize(va["label"], np.array(LABELS)[p.argmax(1)], p[:, 1])
            history.append({"epoch": ep + 1, "train_loss": total / len(tl.dataset)} | s)
            print(f"epoch {ep + 1:3d}  loss {total / len(tl.dataset):.4f}  "
                  f"val challenge F1 {s['challenge_F1']:.3f}")
            if s["challenge_F1"] > best:
                best = s["challenge_F1"]
                torch.save(model.state_dict(), model_path)
        (out_dir / "val").mkdir(parents=True, exist_ok=True)
        json.dump(history, open(out_dir / "val" / "cnn_history.json", "w"), indent=2)
        print(f"best val challenge F1 {best:.3f} -> {model_path}")
    else:
        te_x, te_y, te = subset(cfg, splits, "test")
        model.load_state_dict(torch.load(model_path, map_location=device))
        p = predict_proba(model, DataLoader(ECGSet(te_x, te_y, False), batch_size=bs, num_workers=nw), device)
        pred = np.array(LABELS)[p.argmax(1)]
        (out_dir / "test").mkdir(parents=True, exist_ok=True)
        summary = summarize(te["label"], pred, p[:, 1])
        json.dump(summary, open(out_dir / "test" / "cnn_summary.json", "w"), indent=2)
        pd.DataFrame({"record": te["record"], "label": te["label"], "pred": pred, "p_af": p[:, 1]}).to_csv(
            out_dir / "test" / "cnn_predictions.csv", index=False)
        print(summary)


if __name__ == "__main__":
    main()
