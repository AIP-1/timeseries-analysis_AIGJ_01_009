"""
과제 4 이상 탐지: 오토인코더(정상만 보고 배움) + 규칙 2개(급변, 값 고정)

  ① diagnose.load_clean 으로 같은 정제, 과제 3과 같은 기간 나누기 (학습 7일 / 검증 3일 / 나머지 시험)
     정규화도 같은 방식 (학습 기간 평균·표준편차 = baseline.py 의 norm_stats.csv 와 같은 값)
  ② 오토인코더 학습: 학습 기간(고장 없는 정상 데이터)의 '최근 1시간 모양'을 압축했다가 복원하도록 배움
     → 정상 모양은 잘 복원하고, 처음 보는 모양(고장)은 복원을 못 해서 오차가 커짐
  ③ 경보 임계값: 검증 기간에서 헛경보가 하루 --fa-per-day 회(기본 0.5) 이하가 되는 가장 낮은 값
  ④ 규칙 2개 (기준값은 학습 기간에서 정함)
     - 급변:    5분 사이 변화량이 학습 기간 최대 변화량의 1.2배를 넘으면 경보
     - 값 고정: 30분(6칸) 동안 값이 전혀 안 변하면 경보 (정상 센서는 잡음 때문에 조금씩 흔들림)
  ⑤ 시험 기간 채점 (faults.csv 정답은 여기서만 씀)
     이벤트 탐지율, 탐지 지연, 하루 헛경보 수(전체 공간 합), 고장 종류별 결과, 원인 센서 맞힌 비율

사용:  python3 inject_faults.py           # 연습용: synthetic/ 마지막 1주에 고장을 심어 synthetic_fault/ 생성
       python3 task4_detect.py            # synthetic_fault/ 를 읽고 diagnosis4/ 에 결과 저장
       python3 task4_detect.py --dir 공식데이터폴더   # 공식 데이터 (faults.csv 가 없으면 채점은 건너뜀)
같은 폴더에 diagnose.py, train_task3.py 가 있어야 함
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
from train_task3 import load_schedule      # 일정 → 공간마다 0/1 배열 (과제 3과 같은 함수)

ITEMS = ("co2", "temperature", "humidity")  # 감시하는 센서 항목
DAY = 24 * 60 // BIN                        # 하루 = 288칸
HOUR = 60 // BIN                            # 1시간 = 12칸
WINDOW = HOUR                               # 오토인코더가 한 번에 보는 길이: 최근 1시간(12칸)
MERGE = 6                                   # 경보 칸 사이 빈틈이 30분(6칸) 이하면 경보 하나로 묶음
GRACE = HOUR                                # 고장이 끝난 뒤 1시간 안에 울린 경보까지 '탐지'로 인정
JUMP_RATIO = 1.2                            # 급변 규칙: 학습 기간 최대 변화량의 몇 배부터 경보
FLAT_LEN = 6                                # 값 고정 규칙: 몇 칸(30분) 동안 안 변하면 경보
FLAT_EPS = {"co2": 0.05, "temperature": 0.005, "humidity": 0.05}   # '안 변함'으로 보는 변화폭


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================ ① 데이터 준비
def to_array(series, n):
    """series → raw[공간, 칸, 항목] (원래 단위, 결측 NaN)"""
    raw = np.full((len(SPACES), n, len(ITEMS)), np.nan, dtype=np.float32)
    for s, sp in enumerate(SPACES):
        for i, item in enumerate(ITEMS):
            raw[s, :, i] = [np.nan if v is None else v for v in series[(sp, item)]]
    return raw


def context(t0, n, sched_arr):
    """오토인코더에 '힌트'로 주는 정보: 시각 sin·cos, 주말 여부, 최근 1시간 일정, 공간 표시
    힌트는 복원 대상이 아님. "지금은 수업 중이니 CO2가 오르는 게 정상" 같은 판단을 돕는 용도.
    반환: ctx_time[칸, 3], sched_arr[공간, 칸]"""
    times = [t0 + timedelta(minutes=b * BIN) for b in range(n)]
    tod = np.array([(t.hour * 60 + t.minute) / 1440 for t in times], dtype=np.float32)
    weekend = np.array([t.weekday() >= 5 for t in times], dtype=np.float32)
    return np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), weekend], axis=1)


def windows(z, ctx_time, sched_arr, idx):
    """(공간, 칸 t) 목록 → 입력 x(최근 1시간 센서값) 와 힌트 c"""
    s, t = idx[:, 0], idx[:, 1]
    win = t[:, None] + np.arange(-WINDOW + 1, 1)                  # (B, 12)
    x = z[s[:, None], win].reshape(len(idx), -1)                  # (B, 12칸 × 3항목 = 36)
    onehot = np.eye(len(SPACES), dtype=np.float32)[s]
    c = np.concatenate([ctx_time[t], sched_arr[s[:, None], win], onehot], axis=1)   # (B, 3+12+4)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(c.astype(np.float32))


def valid_points(z, start, end):
    """[start, end) 에서 최근 1시간에 결측이 하나도 없는 (공간, t)"""
    ok = ~np.isnan(z).any(axis=2)                                 # (공간, 칸)
    pts = []
    for s in range(z.shape[0]):
        for t in range(max(start, WINDOW - 1), end):
            if ok[s, t - WINDOW + 1: t + 1].all():
                pts.append((s, t))
    return np.array(pts)


# ============================================================ ② 오토인코더
class AutoEncoder(nn.Module):
    """36개 값(1시간 × 3항목)을 4개 숫자로 압축(인코더) → 다시 36개로 복원(디코더)

    4개 숫자에 다 담을 수 없으니 '정상일 때 자주 보이는 모양'만 기억하게 됨.
    힌트(시각·일정·공간)는 인코더와 디코더 양쪽에 같이 넣어 줌.
    """
    def __init__(self, n_in, n_ctx, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in + n_ctx, 64), nn.ReLU(),
                                 nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, latent))
        self.dec = nn.Sequential(nn.Linear(latent + n_ctx, 16), nn.ReLU(),
                                 nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, n_in))

    def forward(self, x, c):
        code = self.enc(torch.cat([x, c], dim=1))
        return self.dec(torch.cat([code, c], dim=1))


def train_ae(z, ctx_time, sched_arr, tr_pts, va_pts, a):
    x0, c0 = windows(z, ctx_time, sched_arr, tr_pts[:1])
    model = AutoEncoder(x0.shape[1], c0.shape[1], a.latent)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    xv, cv = windows(z, ctx_time, sched_arr, va_pts)
    best, best_state, wait = float("inf"), None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr_pts))
        tot = 0.0
        for i in range(0, len(order), a.batch):
            x, c = windows(z, ctx_time, sched_arr, tr_pts[order[i:i + a.batch]])
            loss = ((model(x, c) - x) ** 2).mean()       # 복원 오차 (입력과 출력이 얼마나 다른가)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            vl = ((model(xv, cv) - xv) ** 2).mean().item()
        mark = ""
        if vl < best:
            best, wait, mark = vl, 0, " ← 최고"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 5 == 0 or mark:
            print(f"    epoch {ep:3}  학습 {tot / len(tr_pts):.4f}  검증 {vl:.4f}{mark}")
        if wait >= a.patience:
            print(f"    검증 오차가 {a.patience}번 연속 안 좋아져서 멈춤 (epoch {ep})")
            break
    model.load_state_dict(best_state)
    return model


def item_errors(model, z, ctx_time, sched_arr, pts):
    """각 (공간, t)의 항목별 복원 오차 = 최근 1시간 동안 (복원값 - 실제값)² 의 평균"""
    out = np.full((z.shape[0], z.shape[1], len(ITEMS)), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pts), 4096):
            p = pts[i:i + 4096]
            x, c = windows(z, ctx_time, sched_arr, p)
            e = ((model(x, c) - x) ** 2).numpy().reshape(len(p), WINDOW, len(ITEMS)).mean(axis=1)
            out[p[:, 0], p[:, 1]] = e
    return out


# ============================================================ 경보 묶기 · 세기
def episodes(flag):
    """한 공간의 칸별 경보(True/False) → 경보 목록 [(시작 칸, 끝 칸), ...]
    빈틈이 MERGE 칸 이하면 같은 경보로 봄 (같은 고장으로 여러 번 울리는 것을 하나로 셈)"""
    eps = []
    for t in np.flatnonzero(flag).tolist():
        if eps and t - eps[-1][1] <= MERGE:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


def count_alarms(flags, start, end):
    """[start, end) 기간에 시작한 경보 수 (모든 공간 합)"""
    return sum(start <= e[0] < end for s in range(flags.shape[0]) for e in episodes(flags[s]))


def pick_threshold(score, start, end, per_day):
    """검증 기간에서 경보가 하루 per_day 회 이하가 되는 가장 낮은 임계값
    (검증 기간은 정상이라고 보므로 여기서 울리는 경보는 모두 헛경보)"""
    days = (end - start) / DAY
    vals = score[:, start:end]
    vals = np.sort(vals[~np.isnan(vals)])
    cands = np.unique(np.quantile(vals, np.linspace(0.9, 1.0, 1001)))
    for th in cands:                                      # 낮은 값부터 올려 가며 확인
        flags = np.nan_to_num(score, nan=0) > th
        if count_alarms(flags, start, end) <= per_day * days:
            return th
    return vals[-1]


# ============================================================ ④ 규칙
def rule_jump(raw, tr_end):
    """급변: 5분 사이 변화량 > 학습 기간 최대 변화량 × 1.2 (공간·항목마다 따로)
    반환: 칸별 경보[공간, 칸], 어느 항목 때문인지[공간, 칸], 기준값[공간, 항목]"""
    d = np.abs(np.diff(raw, axis=1, prepend=np.nan))       # 바로 앞 칸과의 차이
    limit = np.nanmax(d[:, :tr_end], axis=1) * JUMP_RATIO  # (공간, 항목)
    ratio = np.nan_to_num(d / limit[:, None, :], nan=0)
    return ratio.max(axis=2) > 1, ratio.argmax(axis=2), limit


def rule_flat(raw):
    """값 고정: 최근 FLAT_LEN 칸(30분) 동안 값의 변화폭(최대 - 최소)이 거의 0"""
    S, N, I = raw.shape
    flag = np.zeros((S, N), dtype=bool)
    which = np.zeros((S, N), dtype=int)
    eps = np.array([FLAT_EPS[it] for it in ITEMS])
    for t in range(FLAT_LEN - 1, N):
        w = raw[:, t - FLAT_LEN + 1: t + 1]                # (공간, 6칸, 항목)
        full = ~np.isnan(w).any(axis=1)                    # 6칸이 다 있어야 판단
        rng = np.where(full, np.ptp(np.nan_to_num(w), axis=1), np.inf)   # 변화폭 = 최대 - 최소
        stuck = rng <= eps
        flag[:, t] = stuck.any(axis=1)
        which[:, t] = stuck.argmax(axis=1)
    return flag, which


# ============================================================ ⑤ 채점
def read_faults(path, t0):
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b0 = int((datetime.fromisoformat(r["start"]) - t0).total_seconds() // 60 // BIN)
            b1 = int((datetime.fromisoformat(r["end"]) - t0).total_seconds() // 60 // BIN)
            out.append(dict(id=r["fault_id"], type=r["type"], space=SPACES.index(r["space"]),
                            item=ITEMS.index(r["item"]), b0=b0, b1=b1))
    return out


def score_method(flags, which, faults, start, end):
    """한 방법의 경보를 정답과 맞춰 봄
    탐지: 같은 공간에서 [고장 시작, 고장 끝 + 1시간] 안에 경보 칸이 있으면 탐지
    지연: 고장 시작 → 그 구간의 첫 경보 칸까지 (분)
    헛경보: 어느 고장 구간과도 겹치지 않는 경보"""
    rows = []
    covered = np.zeros(flags.shape, dtype=bool)            # 고장 구간 표시 (헛경보 판정용)
    for f in faults:
        lo, hi = f["b0"], min(f["b1"] + GRACE, flags.shape[1] - 1)
        covered[f["space"], lo:hi + 1] = True
        hit = np.flatnonzero(flags[f["space"], lo:hi + 1])
        if len(hit):
            first = lo + int(hit[0])
            # 원인 센서: 고장 구간에서 경보가 울린 칸들에서 가장 많이 지목된 항목
            picked = np.bincount(which[f["space"], lo + hit], minlength=len(ITEMS)).argmax()
            rows.append(dict(f, hit=True, delay=(first - f["b0"]) * BIN, cause_ok=picked == f["item"]))
        else:
            rows.append(dict(f, hit=False, delay=None, cause_ok=None))
    false = 0
    for s in range(flags.shape[0]):
        for e0, e1 in episodes(flags[s]):
            if start <= e0 < end and not covered[s, e0:e1 + 1].any():
                false += 1
    return rows, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic_fault")
    ap.add_argument("--out", default="diagnosis4")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--val-days", type=int, default=3)
    ap.add_argument("--fa-per-day", type=float, default=0.5,
                    help="검증 기간 오토인코더 헛경보 목표(하루 몇 회). 규칙 몫을 남기려고 1보다 낮게")
    ap.add_argument("--latent", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)

    # ① 정제 · 기간 · 정규화
    t0, n, series = load_clean(os.path.join(a.dir, "sensor.csv"))
    tr_end = a.train_days * DAY
    va_end = (a.train_days + a.val_days) * DAY
    day = lambda b: (t0 + timedelta(minutes=b * BIN)).date()
    print("① 기간 나누기 (과제 3과 같음)")
    print(f"  학습 {day(0)} ~ {day(tr_end - 1)} / 검증 {day(tr_end)} ~ {day(va_end - 1)}"
          f" / 시험 {day(va_end)} ~ {day(n - 1)}")
    raw = to_array(series, n)
    mean = np.nanmean(raw[:, :tr_end], axis=1)             # 학습 기간으로만 (공간, 항목)
    std = np.nanstd(raw[:, :tr_end], axis=1)
    z = (raw - mean[:, None]) / std[:, None]
    sched = load_schedule(os.path.join(a.dir, a.schedule), t0, n)
    sched_arr = np.stack([sched[sp] for sp in SPACES])
    ctx_time = context(t0, n, sched_arr)

    # ② 오토인코더 학습 (학습 기간만)
    tr_pts = valid_points(z, 0, tr_end)
    va_pts = valid_points(z, tr_end, va_end)
    all_pts = valid_points(z, 0, n)
    print()
    print(f"② 오토인코더 학습  (학습 창 {len(tr_pts)}개, 검증 창 {len(va_pts)}개, 압축 크기 {a.latent})")
    model = train_ae(z, ctx_time, sched_arr, tr_pts, va_pts, a)
    torch.save(model.state_dict(), os.path.join(a.out, "autoencoder.pt"))

    # ③ 이상 점수 · 임계값
    err = item_errors(model, z, ctx_time, sched_arr, all_pts)   # (공간, 칸, 항목)
    # 항목마다 오차 크기가 달라서, 검증 기간 오차의 99% 지점으로 나눠 맞춤 (1 = 평소 큰 편인 오차)
    # 학습 기간 오차는 모델이 외운 데이터라 너무 작게 나옴 → 처음 보는 정상 데이터(검증)로 잼
    unit = np.nanquantile(err[:, tr_end:va_end].reshape(-1, len(ITEMS)), 0.99, axis=0)
    rel = err / unit
    score = np.max(np.nan_to_num(rel, nan=-1), axis=2)    # 이상 점수 = 세 항목 중 가장 큰 값
    score[score < 0] = np.nan                               # 결측이 있어 점수를 못 낸 칸
    ae_which = np.nan_to_num(rel, nan=-1).argmax(axis=2)    # 원인 센서 = 오차가 가장 큰 항목
    th = pick_threshold(score, tr_end, va_end, a.fa_per_day)
    ae_flag = np.nan_to_num(score, nan=0) > th
    print()
    print(f"③ 경보 임계값 = {th:.2f}  (검증 기간 헛경보 하루 {a.fa_per_day}회 이하가 되는 가장 낮은 값)")
    print(f"  점수 뜻: 1 = 검증 기간 정상 오차의 99% 지점. 임계값 {th:.1f} = 평소보다 {th:.1f}배 크게 못 맞추면 경보")

    # ④ 규칙
    jump_flag, jump_which, jump_limit = rule_jump(raw, tr_end)
    flat_flag, flat_which = rule_flat(raw)
    print()
    print("④ 규칙 기준값 (학습 기간에서 정함)")
    print("  급변: 5분 변화량이 아래보다 크면 경보 (학습 기간 최대 변화량 × 1.2)")
    for si, sp in enumerate(SPACES):
        print(f"    {sp:5} CO2 {jump_limit[si, 0]:5.0f}ppm   온도 {jump_limit[si, 1]:.1f}℃   습도 {jump_limit[si, 2]:.1f}%")
    print(f"  값 고정: {FLAT_LEN * BIN}분 동안 변화폭이 CO2 {FLAT_EPS['co2']}ppm, 온도 {FLAT_EPS['temperature']}℃,"
          f" 습도 {FLAT_EPS['humidity']}% 이하")

    # 방법별 경보: 원인 센서 표시는 방법마다 따로
    rules_flag = jump_flag | flat_flag
    rules_which = np.where(jump_flag, jump_which, flat_which)
    all_flag = ae_flag | rules_flag
    all_which = np.where(ae_flag, ae_which, rules_which)
    methods = {"오토인코더": (ae_flag, ae_which), "규칙: 급변": (jump_flag, jump_which),
               "규칙: 값 고정": (flat_flag, flat_which), "규칙 둘": (rules_flag, rules_which),
               "오토인코더+규칙": (all_flag, all_which)}

    print()
    print("  기간별 경보 수 (모든 공간 합, 하루 평균)")
    print(f"  {'':16}{'학습':>8}{'검증':>8}{'시험':>8}")
    for name, (fl, _) in methods.items():
        vals = [count_alarms(fl, s, e) / ((e - s) / DAY) for s, e in ((0, tr_end), (tr_end, va_end), (va_end, n))]
        print(f"  {name:16}" + "".join(f"{v:8.2f}" for v in vals))

    # 경보 목록 저장
    with open(os.path.join(a.out, "alarms.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "space", "start", "end", "cause_item"])
        for name, (fl, wh) in methods.items():
            for s, sp in enumerate(SPACES):
                for e0, e1 in episodes(fl[s]):
                    item = ITEMS[np.bincount(wh[s, e0:e1 + 1][fl[s, e0:e1 + 1]], minlength=3).argmax()]
                    w.writerow([name, sp, (t0 + timedelta(minutes=e0 * BIN)).isoformat(),
                                (t0 + timedelta(minutes=(e1 + 1) * BIN)).isoformat(), item])

    # ⑤ 채점
    fpath = os.path.join(a.dir, "faults.csv")
    if not os.path.exists(fpath):
        print(f"\n  {fpath} 이 없어 채점은 건너뜀. 경보 목록: {a.out}/alarms.csv")
        return
    faults = read_faults(fpath, t0)
    test_days = (n - va_end) / DAY
    types = list(dict.fromkeys(f["type"] for f in faults))
    print()
    print(f"⑤ 시험 기간 채점 (고장 {len(faults)}건, 시험 {test_days:.0f}일)")
    print(f"  {'':16}{'탐지율':>8}{'평균 지연':>10}{'헛경보/일':>10}{'원인 센서':>10}   " +
          "  ".join(f"{t}" for t in types))
    results = {}
    with open(os.path.join(a.out, "task4_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "fault_id", "type", "space", "item", "detected", "delay_min", "cause_ok"])
        for name, (fl, wh) in methods.items():
            rows, false = score_method(fl, wh, faults, va_end, n)
            results[name] = rows
            hits = [r for r in rows if r["hit"]]
            delay = np.mean([r["delay"] for r in hits]) if hits else float("nan")
            cause = np.mean([r["cause_ok"] for r in hits]) * 100 if hits else float("nan")
            by_type = "  ".join(f"{sum(r['hit'] for r in rows if r['type'] == t)}/"
                                f"{sum(r['type'] == t for r in rows)}".center(len(t)) for t in types)
            print(f"  {name:16}{len(hits) / len(rows) * 100:7.0f}%{delay:8.0f}분{false / test_days:10.2f}"
                  f"{cause:9.0f}%   {by_type}")
            for r in rows:
                w.writerow([name, r["id"], r["type"], SPACES[r["space"]], ITEMS[r["item"]],
                            int(r["hit"]), r["delay"], r["cause_ok"]])

    print()
    print("  [고장별 상세: 오토인코더+규칙]  지연(분) / 지목한 원인 센서")
    for r in results["오토인코더+규칙"]:
        src = []
        for name in ("오토인코더", "규칙: 급변", "규칙: 값 고정"):
            rr = next(x for x in results[name] if x["id"] == r["id"])
            if rr["hit"]:
                src.append(f"{name.replace('규칙: ', '')} {rr['delay']}분")
        st = t0 + timedelta(minutes=r["b0"] * BIN)
        mark = ("원인 맞음" if r["cause_ok"] else "원인 틀림") if r["hit"] else "놓침"
        print(f"  {r['id']:>3}. {r['type']:6} {SPACES[r['space']]:4} {ITEMS[r['item']]:11} {st:%m-%d %H:%M}"
              f"  {mark:5}  " + (", ".join(src) if src else "-"))
    print()
    print(f"  저장: {a.out}/autoencoder.pt, alarms.csv, task4_results.csv")


if __name__ == "__main__":
    main()
