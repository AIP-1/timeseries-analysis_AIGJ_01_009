"""
과제 6 사전학습 시계열 모델(Chronos) vs 기준선 vs 직접 학습한 GRU  — 시험 기간 60분 뒤 CO2

  ① baseline.py 와 같은 데이터·기간, eval_points.csv 의 같은 평가 시점(CO2)만 사용
  ② 직접 학습한 GRU 사다리 (입력 정보를 하나씩 늘려서 '무슨 정보 덕분인지' 나눔)
       CO2만 → +시각·주말 → +온습도(= 과제 3 일정 제외) → +일정(= 과제 3 일정 포함)
       과제 3 두 모델은 train_task3.py 가 저장한 gru_*.pt 를 그대로 불러옴
  ③ 사전학습 모델: 우리 데이터로 추가 학습 없이 과거 CO2만 주고 바로 예측
       Chronos-Bolt tiny / small : CO2 과거값만 받음
       Chronos-2                  : CO2만 / +온습도·시각(다른 변수) / +앞으로 60분 일정 도 시험
  ④ 정확도(MAE), 10~90% 예측 구간 적중률(이상적이면 80%), 구간 폭
  ⑤ 과거를 얼마나 길게 줄 때 잘 맞나 (1시간 ~ 7일)
  ⑥ 상황별(빈 방 / 사람 있음 / 입실 / 퇴실 / 노쇼)로 어디서 되고 어디서 안 되나  ← truth.csv 는 채점에만

사용:  python3 baseline.py && python3 train_task3.py     # 과제 3 결과(diagnosis/) 먼저
       python3 task6_compare.py                         # task6_models/ 의 모델을 읽음
       python3 task6_compare.py --quick                 # 평가 시점 1/4만 (빠르게 확인)
결과: diagnosis/task6_results.csv, task6_context.csv, task6_situation.csv
"""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

import train_task3 as T
from diagnose import BIN, SPACES, load_clean

H = 60 // BIN                 # 60분 뒤 = 12칸
Q = [0.1, 0.5, 0.9]           # 10% / 중앙값 / 90%


# ------------------------------------------------------------------ 준비
def load_points(path, t0):
    """eval_points.csv 에서 CO2 평가 시점 (공간 번호, 칸) 목록"""
    sp_idx = {sp: i for i, sp in enumerate(SPACES)}
    pts = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["item"] == "co2":
                t = int((datetime.fromisoformat(r["time"]) - t0).total_seconds() // 60 // BIN)
                pts.add((sp_idx[r["space"]], t))
    return np.array(sorted(pts))


def load_people(path, t0, n):
    """truth.csv → 5분 칸마다 실제로 사람이 있었나 (채점용, 모델 입력에는 절대 안 씀)"""
    sp_idx = {sp: i for i, sp in enumerate(SPACES)}
    ppl = np.zeros((len(SPACES), n))
    cnt = np.zeros((len(SPACES), n))
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = int((datetime.fromisoformat(r["time"]) - t0).total_seconds() // 60 // BIN)
            if 0 <= b < n and r["space"] in sp_idx:
                ppl[sp_idx[r["space"]], b] += float(r["people"])
                cnt[sp_idx[r["space"]], b] += 1
    return (ppl / np.maximum(cnt, 1)) >= 0.5


def situations(pts, occ, sched_arr):
    """평가 시점마다 앞으로 60분이 어떤 상황이었나 (situation_analysis.py 와 같은 나눔)"""
    s, t = pts[:, 0], pts[:, 1]
    fw = t[:, None] + np.arange(1, H + 1)
    pw = t[:, None] + np.arange(-24, 1)
    occ_now, occ_fut, occ_past = occ[s, t], occ[s[:, None], fw], occ[s[:, None], pw].any(1)
    sch_now, sch_fut = sched_arr[s, t] > 0, sched_arr[s[:, None], fw]
    enter = (~occ_now) & occ_fut.any(1)
    leave = occ_now & (~occ_fut).any(1)
    noshow = (~sch_now) & (sch_fut.max(1) > 0) & (~occ_fut.any(1))
    ph = np.full(len(pts), "기타", object)
    ph[(~occ_now) & (~occ_fut.any(1)) & (~occ_past)] = "A 빈 방 계속"
    ph[(~occ_now) & (~occ_fut.any(1)) & occ_past] = "B 퇴실 뒤 빠지는 중"
    ph[occ_now & occ_fut.all(1)] = "C 사람 계속 있음"
    ph[enter] = "D 60분 안 입실"
    ph[leave & ~enter] = "E 60분 안 퇴실"
    ph[noshow] = "F 일정 있는데 노쇼"
    return ph


# ------------------------------------------------------------------ GRU
# train_task3.build_features 의 특성 순서:
#   0 co2, 1 온도, 2 습도, 3~5 결측표시, 6 sin, 7 cos, 8 주말, (9 일정), 끝 4개 = 공간 표시
LADDER = [
    ("GRU CO2만", [0, 3]),
    ("GRU CO2+시각", [0, 3, 6, 7, 8]),
]


def gru_predict(model, feats, raw, sched_arr, norm_std, pts, use_sched):
    """60분 뒤 CO2 예측값 (ppm)"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), 2048):
            x, fut, _ = T.make_batch(feats, raw, sched_arr, norm_std, pts[i:i + 2048], use_sched)
            out.append(model(x, fut).numpy()[:, 3])        # 출력 3번 = CO2 60분 뒤
    return raw[pts[:, 0], pts[:, 1], 0] + np.concatenate(out) * norm_std[pts[:, 0], 0]


def run_grus(a, t0, n, series, norm, norm_std, sched, sched_arr, pts):
    preds = {}
    tr_end, va_end = a.train_days * T.DAY, (a.train_days + a.val_days) * T.DAY
    fe_ns, raw = T.build_features(t0, n, series, norm, sched, False)
    tr_pts, va_pts = T.sample_points(raw, 0, tr_end), T.sample_points(raw, tr_end, va_end)
    for name, cols in LADDER:
        path = os.path.join(a.out, f"gru_{'co2only' if name.endswith('CO2만') else 'co2time'}.pt")
        sel = cols + list(range(fe_ns.shape[2] - len(SPACES), fe_ns.shape[2]))   # 공간 표시는 둠
        fe = np.ascontiguousarray(fe_ns[:, :, sel])
        T.set_seed(a.seed)
        if os.path.exists(path) and not a.retrain:
            model = T.GRUForecaster(fe.shape[2], 0, a.hidden)
            model.load_state_dict(torch.load(path))
            print(f"  {name}: 저장된 모델 불러옴 ({path})")
        else:
            print(f"  {name} 학습 (입력 특성 {fe.shape[2]}개)")
            model = T.train_one(fe, raw, sched_arr, norm_std, tr_pts, va_pts, False, a)
            torch.save(model.state_dict(), path)
        preds[name] = gru_predict(model, fe, raw, sched_arr, norm_std, pts, False)
    for name, us, fn in (("GRU +온습도 (과제3 일정 제외)", False, "gru_nosched.pt"),
                         ("GRU +일정 (과제3 일정 포함)", True, "gru_sched.pt")):
        fe, _ = T.build_features(t0, n, series, norm, sched, us)
        model = T.GRUForecaster(fe.shape[2], T.FUTURE if us else 0, a.hidden)
        model.load_state_dict(torch.load(os.path.join(a.out, fn)))
        preds[name] = gru_predict(model, fe, raw, sched_arr, norm_std, pts, us)
    return preds, raw


# ------------------------------------------------------------------ Chronos
def contexts(co2, pts, L):
    """평가 시점마다 지난 L칸의 CO2 (모자라거나 비어 있으면 NaN → 모델이 알아서 가림)"""
    idx = pts[:, 1][:, None] + np.arange(-L + 1, 1)
    ctx = co2[pts[:, 0][:, None], np.clip(idx, 0, None)]
    ctx[idx < 0] = np.nan
    return ctx.astype(np.float32)


def covariates(t0, n, series, sched_arr):
    """Chronos-2 에 줄 다른 변수: 온도·습도(과거만), 시각 sin·cos·주말·일정(미래도 미리 앎)"""
    tod = np.array([((b * BIN) % 1440) / 1440 for b in range(n)], dtype=np.float32)
    wd = np.array([((t0.weekday() * 1440 + b * BIN) // 1440) % 7 >= 5 for b in range(n)], dtype=np.float32)
    cov = {}
    for si, sp in enumerate(SPACES):
        cov[si] = {
            "temperature": np.array([np.nan if v is None else v for v in series[(sp, "temperature")]], np.float32),
            "humidity": np.array([np.nan if v is None else v for v in series[(sp, "humidity")]], np.float32),
            "tod_sin": np.sin(2 * np.pi * tod), "tod_cos": np.cos(2 * np.pi * tod), "weekend": wd,
            "schedule": sched_arr[si].astype(np.float32),
        }
    return cov


def fill(v):
    """다른 변수의 빈칸: 앞 값으로 채우고, 맨 앞이 비면 첫 값으로 (전부 비면 0)"""
    if not np.isnan(v).any():
        return v
    if np.isnan(v).all():
        return np.zeros_like(v)
    v = v.copy()
    first = int(np.argmax(~np.isnan(v)))
    v[:first] = v[first]
    return T.forward_fill(v).astype(np.float32)


def chronos_predict(pipe, co2, pts, L, batch, cov=None, use_sched=False):
    """(N, 3) = 60분 뒤 10% / 50% / 90%"""
    is2 = type(pipe).__name__ == "Chronos2Pipeline"
    out = []
    for i in range(0, len(pts), batch):
        p = pts[i:i + batch]
        ctx = contexts(co2, p, L)
        if cov is None:
            x = torch.from_numpy(ctx[:, None, :] if is2 else ctx)
        else:
            x = []
            for (s, t), c in zip(p, ctx):
                keys = [k for k in cov[s] if use_sched or k != "schedule"]
                lo = t - L + 1
                past = {}
                for k in keys:
                    v = cov[s][k][max(0, lo): t + 1]
                    v = np.concatenate([np.full(len(c) - len(v), v[0] if len(v) else 0, np.float32), v])
                    past[k] = fill(v.astype(np.float32))
                fut = {k: cov[s][k][t + 1: t + 1 + H] for k in keys if k not in ("temperature", "humidity")}
                x.append({"target": c, "past_covariates": past, "future_covariates": fut})
        q, _ = pipe.predict_quantiles(x, prediction_length=H, quantile_levels=Q)
        if isinstance(q, (list, tuple)):
            q = torch.stack([qq.reshape(-1, H, len(Q))[0] for qq in q])
        out.append(q[:, -1, :].float().numpy())
    return np.concatenate(out)


def load_pipe(path):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(path, device_map="cpu", torch_dtype=torch.float32)


# ------------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic")
    ap.add_argument("--out", default="diagnosis")
    ap.add_argument("--models", default="task6_models")
    ap.add_argument("--chronos", default="chronos-bolt-tiny,chronos-bolt-small,chronos-2")
    ap.add_argument("--context", type=int, default=2016, help="본 비교에 주는 과거 칸 수 (2016칸 = 7일)")
    ap.add_argument("--sweep", default="12,24,72,288,864,2016", help="과거 길이 시험 (칸)")
    ap.add_argument("--sweep-models", default="chronos-bolt-small,chronos-2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cov-every", type=int, default=12, help="다른 변수를 넣은 Chronos-2 는 평가 시점 N개 중 1개만 (느림)")
    ap.add_argument("--sweep-every", type=int, default=4, help="과거 길이 시험은 평가 시점 N개 중 1개만")
    ap.add_argument("--quick", action="store_true", help="평가 시점 1/4만")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--val-days", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.batch_train = 256
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    t0, n, series = load_clean(os.path.join(a.dir, "sensor.csv"))
    norm = T.read_norm(os.path.join(a.out, "norm_stats.csv"))
    norm_std = np.array([[norm[(sp, it)][1] for it in T.ITEMS] for sp in SPACES], dtype=np.float32)
    sched = T.load_schedule(os.path.join(a.dir, "schedule.csv"), t0, n)
    sched_arr = np.stack([sched[sp] for sp in SPACES])
    pts = load_points(os.path.join(a.out, "eval_points.csv"), t0)
    if a.quick:
        pts = pts[::4]
    print(f"① 시험 기간 CO2 평가 시점 {len(pts)}개 (eval_points.csv 와 같음, 공간 {len(SPACES)}곳)")

    print("② 직접 학습한 GRU (입력 정보 사다리)")
    batch_pred = a.batch
    a.batch = a.batch_train
    preds, raw = run_grus(a, t0, n, series, norm, norm_std, sched, sched_arr, pts)
    a.batch = batch_pred
    co2 = raw[:, :, 0]
    now, true = co2[pts[:, 0], pts[:, 1]], co2[pts[:, 0], pts[:, 1] + H]
    preds = {"현재값 유지 (기준선)": now.copy(), **preds}
    quant = {}

    print("③ 사전학습 모델 (추가 학습 없음)")
    cov = covariates(t0, n, series, sched_arr)
    sub = np.zeros(len(pts), bool)
    sub[::a.cov_every] = True              # 다른 변수까지 넣는 Chronos-2 는 느려서 일부 시점만
    runs = []
    for m in a.chronos.split(","):
        runs.append((m, f"{m} (CO2만)", None, False))
        if m == "chronos-2":
            runs += [(m, f"{m} +온습도·시각", cov, False), (m, f"{m} +온습도·시각·일정", cov, True)]
    pipes = {}
    for m, label, cv, us in runs:
        path = os.path.join(a.models, m)
        if not os.path.isdir(path):
            print(f"  ※ {path} 가 없어 {label} 은 건너뜀")
            continue
        pipes.setdefault(m, load_pipe(path))
        tic = time.perf_counter()
        msk = sub if cv is not None else np.ones(len(pts), bool)
        q = np.full((len(pts), len(Q)), np.nan, np.float32)
        q[msk] = chronos_predict(pipes[m], co2, pts[msk], a.context, a.batch, cv, us)
        quant[label] = q
        preds[label] = q[:, 1]
        print(f"  {label:28} 과거 {a.context}칸, 시점 {msk.sum()}개  {time.perf_counter() - tic:6.1f}초")
    np.savez(os.path.join(a.out, "task6_preds.npz"), pts=pts, true=true,
             **{f"pred::{k}": v for k, v in preds.items()}, **{f"quant::{k}": v for k, v in quant.items()})

    ph = situations(pts, load_people(os.path.join(a.dir, "truth.csv"), t0, n), sched_arr)
    ok = ~np.isnan(true)

    def hit(k, m_):
        q = quant[k]
        return np.mean((true[m_] >= q[m_, 0]) & (true[m_] <= q[m_, 2])) * 100

    def table(title, mask, fname, keys):
        print()
        print(title)
        print(f"  {'방법':32}{'MAE':>8}{'10~90% 적중':>12}{'구간 폭':>9}" + "".join(f"{sp:>8}" for sp in SPACES))
        rows = []
        for k in keys:
            m_ = ok & mask
            e = np.abs(preds[k] - true)
            cov_s = f"{hit(k, m_):.1f}%" if k in quant else ""
            wid = f"{np.mean(quant[k][m_, 2] - quant[k][m_, 0]):.0f}" if k in quant else ""
            per = [np.mean(e[m_ & (pts[:, 0] == si)]) for si in range(len(SPACES))]
            print(f"  {k:32}{np.mean(e[m_]):8.1f}{cov_s:>12}{wid:>9}" + "".join(f"{v:8.1f}" for v in per))
            rows.append([k, round(float(np.mean(e[m_])), 2), cov_s, wid] + [round(float(v), 2) for v in per]
                        + [int(m_.sum())])
        with open(os.path.join(a.out, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "mae_60min_co2", "coverage_10_90", "interval_width"] + SPACES + ["n"])
            w.writerows(rows)

    full_keys = [k for k in preds if not np.isnan(preds[k]).any()]
    table(f"④ 시험 기간 60분 뒤 CO2 — 모든 평가 시점 {ok.sum()}개", np.ones(len(pts), bool),
          "task6_results.csv", full_keys)
    if any(np.isnan(preds[k]).any() for k in preds):
        table(f"④-2 다른 변수를 넣은 Chronos-2 포함 — 같은 일부 시점 {(ok & sub).sum()}개 (평가 시점 {a.cov_every}개 중 1개)",
              sub, "task6_results_sub.csv", list(preds))

    # ⑥ 상황별 (일부 시점이면 그 시점 기준)
    print()
    print("⑥ 상황별 60분 뒤 CO2 MAE (괄호: 10~90% 적중률)  ※ 상황은 truth.csv 로 나눔(채점에만 씀)")
    keys = list(preds)
    print(f"  {'상황':18}{'비율':>7}" + "".join(f"{k[:14]:>16}" for k in keys))
    srows = []
    for g in sorted(set(ph)):
        cells = []
        for k in keys:
            m_ = ok & (ph == g) & ~np.isnan(preds[k])
            e = np.abs(preds[k] - true)[m_].mean()
            c = f"({hit(k, m_):.0f}%)" if k in quant else ""
            cells.append(f"{e:.1f}{c}")
            srows.append([g, round((ok & (ph == g)).sum() / ok.sum() * 100, 1), k, round(float(e), 2),
                          c.strip("()"), int(m_.sum())])
        print(f"  {g:18}{(ok & (ph == g)).sum() / ok.sum() * 100:6.1f}%" + "".join(f"{c:>16}" for c in cells))
    with open(os.path.join(a.out, "task6_situation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["situation", "share_pct", "method", "mae", "coverage", "n"])
        w.writerows(srows)

    # ⑤ 과거 길이
    print()
    sw = np.zeros(len(pts), bool)
    sw[::a.sweep_every] = True
    print(f"⑤ 과거를 얼마나 길게 줄 때 잘 맞나 (CO2만, 60분 뒤 MAE / 10~90% 적중률, 평가 시점 {sw.sum()}개)")
    sweep = [int(x) for x in a.sweep.split(",")]
    print(f"  {'모델':22}" + "".join(f"{f'{L * BIN / 60:g}시간':>14}" for L in sweep))
    crow = []
    for m in a.sweep_models.split(","):
        if m not in pipes:
            continue
        cells = []
        for L in sweep:
            q = chronos_predict(pipes[m], co2, pts[sw], L, a.batch)
            o = ok[sw]
            tr = true[sw]
            mae = np.mean(np.abs(q[o, 1] - tr[o]))
            cv = np.mean((tr[o] >= q[o, 0]) & (tr[o] <= q[o, 2])) * 100
            cells.append(f"{mae:.1f}/{cv:.0f}%")
            crow.append([m, L, L * BIN / 60, round(float(mae), 2), round(float(cv), 1), int(o.sum())])
        print(f"  {m:22}" + "".join(f"{c:>14}" for c in cells))
    with open(os.path.join(a.out, "task6_context.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "context_bins", "context_hours", "mae", "coverage", "n"])
        w.writerows(crow)
    print()
    print(f"  저장: {a.out}/task6_results*.csv, task6_situation.csv, task6_context.csv, task6_preds.npz")


if __name__ == "__main__":
    main()
