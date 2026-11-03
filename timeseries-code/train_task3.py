"""
과제 3 GRU로 5·15·30·60분 뒤 CO2·온도 예측 (baseline.py 와 같은 데이터·기간·평가 시점 사용)

  ① diagnose.load_clean 으로 같은 정제, baseline.py 와 같은 기간 나누기 (7일 / 3일 / 나머지)
  ② 정규화는 baseline.py 가 만든 norm_stats.csv (학습 기간 값)를 그대로 사용
  ③ GRU 학습: 지난 2시간을 보고 4개 시점 × 2항목(CO2·온도)을 한 번에 예측
     - 일정 포함 모델: 수업·예약 일정(앞으로 60분 일정까지)도 입력으로 넣음
     - 일정 제외 모델: 센서값과 시각만 넣음  → 두 모델을 비교
  ④ eval_points.csv 의 같은 평가 시점에서 MAE를 계산해 baseline_results.csv 와 나란히 비교
     → model_results.csv

사용:  python3 baseline.py          # 먼저 실행해서 diagnosis/ 를 만들어 둬야 함
       python3 train_task3.py       # synthetic/ 와 diagnosis/ 를 읽고 diagnosis/ 에 저장
       python3 train_task3.py --epochs 10 --no-compare   # 일정 포함 모델만 빠르게
같은 폴더에 diagnose.py 가 있어야 함
"""
import argparse
import csv
import os
import random
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

from diagnose import BIN, SPACES, load_clean

ITEMS = ("co2", "temperature")          # 예측할 항목
INPUTS = ("co2", "temperature", "humidity")  # 입력으로 쓰는 센서 항목
HORIZONS = (5, 15, 30, 60)              # 몇 분 뒤
DAY = 24 * 60 // BIN                    # 하루 = 288칸
HOUR = 60 // BIN                        # 1시간 = 12칸
WINDOW = 2 * HOUR                       # 지난 2시간(24칸)을 보고 예측
FUTURE = max(HORIZONS) // BIN           # 앞으로 60분(12칸)의 일정을 미리 봄


def set_seed(seed):
    """실행할 때마다 같은 결과가 나오도록 난수 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_norm(path):
    """norm_stats.csv → {(공간, 항목): (평균, 표준편차)}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["space"], r["item"])] = (float(r["mean"]), float(r["std"]) or 1.0)
    return out


def load_schedule(path, t0, n):
    """일정 파일 → 공간마다 길이 n 의 0/1 배열 (그 칸에 일정이 잡혀 있으면 1)

    일정은 '계획'일 뿐이라 실제 사용과 다를 수 있음 (노쇼, 연장 등).
    그래도 미래 일정은 미리 알 수 있는 정보라서 60분 뒤 예측에 특히 도움이 됨.
    컬럼 이름은 space / start / end 를 기본으로 찾고, 비슷한 이름도 받아 줌.
    """
    sched = {sp: np.zeros(n, dtype=np.float32) for sp in SPACES}
    if not os.path.exists(path):
        print(f"  ※ 일정 파일 {path} 이 없어 일정은 모두 0으로 둠")
        return sched
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return sched
    cols = rows[0].keys()
    pick = lambda *names: next(c for c in cols if c in names)
    c_sp = pick("space", "space_id", "room")
    c_st = pick("start", "start_time", "begin")
    c_en = pick("end", "end_time", "finish")
    unknown = set()
    for r in rows:
        sp = r[c_sp]
        if sp not in sched:
            unknown.add(sp)
            continue
        st = datetime.fromisoformat(r[c_st])
        en = datetime.fromisoformat(r[c_en])
        i0 = max(0, int((st - t0).total_seconds() // 60 // BIN))
        i1 = min(n, int(-(-(en - t0).total_seconds() // 60 // BIN)))  # 끝은 올림
        sched[sp][i0:i1] = 1.0
    if unknown:
        print(f"  ※ SPACES 에 없는 공간 이름이라 무시함: {sorted(unknown)}")
    return sched


def build_features(t0, n, series, norm, sched, use_schedule):
    """공간마다 (n, 특성 수) 배열을 만듦. 모든 값은 정규화된 숫자.

    한 칸의 특성:
      센서 3개(정규화) + 결측 표시 3개 + 시각 sin·cos + 주말 여부
      + (일정 포함이면) 지금 일정 여부 + 공간 표시(one-hot 4개)
    """
    minutes = np.array([(t0 + timedelta(minutes=b * BIN)) for b in range(n)])
    tod = np.array([(m.hour * 60 + m.minute) / 1440 for m in minutes], dtype=np.float32)
    weekend = np.array([m.weekday() >= 5 for m in minutes], dtype=np.float32)

    feats, raw = [], []
    for si, sp in enumerate(SPACES):
        cols = []
        masks = []
        for item in INPUTS:
            v = np.array([np.nan if x is None else x for x in series[(sp, item)]], dtype=np.float32)
            m, sd = norm[(sp, item)]
            z = (v - m) / sd
            miss = np.isnan(z).astype(np.float32)
            # 결측은 바로 앞 값으로 채움 (맨 앞이 비면 0 = 평균)
            z = forward_fill(z)
            cols.append(z)
            masks.append(miss)
        f = cols + masks + [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), weekend]
        if use_schedule:
            f.append(sched[sp])
        onehot = np.zeros((len(SPACES), n), dtype=np.float32)
        onehot[si] = 1.0
        feats.append(np.concatenate([np.stack(f), onehot]).T)   # (n, F)
        # 정답 계산용 원래 값 (정규화 전, 결측은 NaN)
        raw.append(np.stack([np.array([np.nan if x is None else x for x in series[(sp, it)]],
                                      dtype=np.float32) for it in ITEMS]).T)   # (n, 2)
    return np.stack(feats), np.stack(raw)   # (공간, n, F), (공간, n, 2)


def forward_fill(a):
    out = a.copy()
    last = 0.0
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


class GRUForecaster(nn.Module):
    """GRU 로 지난 2시간을 요약 → (일정 포함이면) 앞으로 60분 일정을 붙여 → 8개 값 예측

    출력은 '지금 값에서 얼마나 바뀔지'(정규화 단위). 현재값 유지 기준선 위에
    변화량만 배우게 해서 학습이 쉬워짐.
    """
    def __init__(self, n_feat, n_future, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_future, 64), nn.ReLU(),
            nn.Linear(64, len(ITEMS) * len(HORIZONS)))

    def forward(self, x, fut):
        _, h = self.gru(x)              # h[-1]: 마지막 층의 마지막 상태 = 지난 2시간 요약
        z = torch.cat([h[-1], fut], dim=1)
        return self.head(z)             # (배치, 8) = 항목 2 × 시점 4


def make_batch(feats, raw, sched_arr, norm_std, idx, use_schedule):
    """(공간 번호, 칸 t) 목록 → 모델 입력과 정답"""
    sp, t = idx[:, 0], idx[:, 1]
    win = t[:, None] + np.arange(-WINDOW + 1, 1)                 # (B, 24)
    x = feats[sp[:, None], win]                                  # (B, 24, F)
    if use_schedule:
        fwin = t[:, None] + np.arange(1, FUTURE + 1)             # 앞으로 12칸
        fut = sched_arr[sp[:, None], fwin]                       # (B, 12)
    else:
        fut = np.zeros((len(t), 0), dtype=np.float32)
    # 정답: (미래 값 - 지금 값) / 표준편차, 항목 2 × 시점 4
    now = raw[sp, t]                                             # (B, 2)
    ys = []
    for i in range(len(ITEMS)):
        for hm in HORIZONS:
            ys.append((raw[sp, t + hm // BIN, i] - now[:, i]) / norm_std[sp, i])
    y = np.stack(ys, axis=1)                                     # (B, 8), 결측은 NaN
    return (torch.from_numpy(x), torch.from_numpy(fut.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)))


def sample_points(raw, start, end):
    """[start, end) 안에서 지금 값이 있고 60분 뒤까지 데이터 안에 있는 (공간, t)"""
    pts = []
    for s in range(raw.shape[0]):
        for t in range(max(start, WINDOW - 1), end - FUTURE):
            if not np.isnan(raw[s, t]).any():
                pts.append((s, t))
    return np.array(pts)


def train_one(feats, raw, sched_arr, norm_std, tr_pts, va_pts, use_schedule, a):
    n_feat = feats.shape[2]
    model = GRUForecaster(n_feat, FUTURE if use_schedule else 0, a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_state, wait = float("inf"), None, 0

    def loss_of(pred, y):
        # 정답이 비어 있는 칸(NaN)은 빼고 평균 절대 오차 (평가 지표 MAE 와 맞춤)
        ok = ~torch.isnan(y)
        return (pred[ok] - y[ok]).abs().mean()

    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr_pts))
        tot = 0.0
        for i in range(0, len(order), a.batch):
            x, fut, y = make_batch(feats, raw, sched_arr, norm_std, tr_pts[order[i:i + a.batch]], use_schedule)
            loss = loss_of(model(x, fut), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        # 검증 기간 손실로 '언제 멈출지' 정함 (시험 기간은 마지막에 한 번만 봄)
        model.eval()
        with torch.no_grad():
            vl, cnt = 0.0, 0
            for i in range(0, len(va_pts), 2048):
                x, fut, y = make_batch(feats, raw, sched_arr, norm_std, va_pts[i:i + 2048], use_schedule)
                vl += loss_of(model(x, fut), y).item() * len(x)
                cnt += len(x)
            vl /= cnt
        mark = ""
        if vl < best:
            best, wait, mark = vl, 0, " ← 최고"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"    epoch {ep:2}  학습 {tot / len(tr_pts):.4f}  검증 {vl:.4f}{mark}")
        if wait >= a.patience:
            print(f"    검증 손실이 {a.patience}번 연속 안 좋아져서 멈춤")
            break
    model.load_state_dict(best_state)
    return model


def evaluate(model, feats, raw, sched_arr, norm_std, points, use_schedule):
    """eval_points 의 각 (공간, 항목, t)에서 예측 → 오차 목록"""
    model.eval()
    # 같은 (공간, t)는 한 번만 예측하고 CO2·온도 결과를 나눠 씀
    uniq = sorted({(s, t) for s, _, t in points})
    pred = {}
    with torch.no_grad():
        for i in range(0, len(uniq), 2048):
            idx = np.array(uniq[i:i + 2048])
            x, fut, _ = make_batch(feats, raw, sched_arr, norm_std, idx, use_schedule)
            out = model(x, fut).numpy()
            for (s, t), o in zip(uniq[i:i + 2048], out):
                pred[(s, t)] = o
    err = {}
    for s, item, t in points:
        ii = ITEMS.index(item)
        o = pred[(s, t)]
        for hi, hm in enumerate(HORIZONS):
            # 변화량 예측 → 실제 값으로 되돌림: 지금 값 + 변화량 × 표준편차
            p = raw[s, t, ii] + o[ii * len(HORIZONS) + hi] * norm_std[s, ii]
            e = abs(p - raw[s, t + hm // BIN, ii])
            err.setdefault((item, hm), []).append(e)
            err.setdefault((SPACES[s], item, hm), []).append(e)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic")
    ap.add_argument("--out", default="diagnosis")
    ap.add_argument("--schedule", default="schedule.csv", help="--dir 안의 일정 파일 이름")
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--val-days", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compare", action="store_true", help="일정 제외 모델은 학습하지 않음")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # ① baseline.py 와 같은 정제·기간
    t0, n, series = load_clean(os.path.join(a.dir, "sensor.csv"))
    tr_end = a.train_days * DAY
    va_end = (a.train_days + a.val_days) * DAY
    day = lambda b: (t0 + timedelta(minutes=b * BIN)).date()
    print("① 기간 나누기 (baseline.py 와 같음)")
    print(f"  학습 {day(0)} ~ {day(tr_end - 1)} / 검증 {day(tr_end)} ~ {day(va_end - 1)} / 시험 {day(va_end)} ~ {day(n - 1)}")

    # ② 정규화 값: baseline.py 가 학습 기간으로 계산해 둔 것
    norm = read_norm(os.path.join(a.out, "norm_stats.csv"))
    norm_std = np.array([[norm[(sp, it)][1] for it in ITEMS] for sp in SPACES], dtype=np.float32)
    sched = load_schedule(os.path.join(a.dir, a.schedule), t0, n)
    sched_arr = np.stack([sched[sp] for sp in SPACES])
    print(f"② norm_stats.csv 읽음, 일정이 잡힌 칸 비율 "
          + ", ".join(f"{sp} {sched[sp].mean() * 100:.0f}%" for sp in SPACES))

    # 평가 시점: baseline.py 가 저장한 것 그대로
    sp_idx = {sp: i for i, sp in enumerate(SPACES)}
    points = []
    with open(os.path.join(a.out, "eval_points.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = int((datetime.fromisoformat(r["time"]) - t0).total_seconds() // 60 // BIN)
            points.append((sp_idx[r["space"]], r["item"], t))

    variants = [("GRU (일정 포함)", True)] + ([] if a.no_compare else [("GRU (일정 제외)", False)])
    results = {}
    for name, use_sched in variants:
        set_seed(a.seed)
        feats, raw = build_features(t0, n, series, norm, sched, use_sched)
        tr_pts = sample_points(raw, 0, tr_end)
        va_pts = sample_points(raw, tr_end, va_end)
        print()
        print(f"③ {name} 학습  (학습 표본 {len(tr_pts)}개, 검증 표본 {len(va_pts)}개, 입력 특성 {feats.shape[2]}개)")
        model = train_one(feats, raw, sched_arr, norm_std, tr_pts, va_pts, use_sched, a)
        torch.save(model.state_dict(), os.path.join(a.out, f"gru_{'sched' if use_sched else 'nosched'}.pt"))
        results[name] = evaluate(model, feats, raw, sched_arr, norm_std, points, use_sched)

    # ④ 기준선과 나란히 비교
    base = {}
    with open(os.path.join(a.out, "baseline_results.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            base[(r["space"], r["item"], int(r["horizon_min"]), r["baseline"])] = float(r["mae"])
    base_names = list(dict.fromkeys(k[3] for k in base))

    print()
    print("④ 시험 기간 MAE (baseline.py 와 같은 평가 시점)")
    with open(os.path.join(a.out, "model_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["space", "item", "horizon_min", "model", "mae", "n"])
        for item, unit in (("co2", "ppm"), ("temperature", "℃")):
            fmt = "{:10.1f}" if item == "co2" else "{:10.2f}"
            print(f"  [{item} ({unit})]      " + "".join(f"{f'{hm}분 뒤':>10}" for hm in HORIZONS))
            for b in base_names:
                print(f"  {b:14}" + "".join(fmt.format(base[("전체", item, hm, b)]) for hm in HORIZONS))
            for name, err in results.items():
                vals = [float(np.mean(err[(item, hm)])) for hm in HORIZONS]
                print(f"  {name:14}" + "".join(fmt.format(v) for v in vals))
                for hm, v in zip(HORIZONS, vals):
                    w.writerow(["전체", item, hm, name, round(v, 3), len(err[(item, hm)])])
            print()
        print("  [공간별 60분 뒤 CO2 MAE (ppm)]")
        print(f"  {'':14}" + "".join(f"{sp:>8}" for sp in SPACES))
        for b in base_names:
            print(f"  {b:14}" + "".join(f"{base[(sp, 'co2', 60, b)]:8.1f}" for sp in SPACES))
        for name, err in results.items():
            print(f"  {name:14}" + "".join(f"{np.mean(err[(sp, 'co2', 60)]):8.1f}" for sp in SPACES))
            for sp in SPACES:
                for item in ITEMS:
                    for hm in HORIZONS:
                        e = err[(sp, item, hm)]
                        w.writerow([sp, item, hm, name, round(float(np.mean(e)), 3), len(e)])
    print()
    print(f"  저장: {a.out}/model_results.csv, gru_*.pt")


if __name__ == "__main__":
    main()
