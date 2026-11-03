"""과제 3 오차를 상황별로 쪼개고, 선형 모델과 비교"""
import csv, os, sys
from datetime import datetime, timedelta
import numpy as np, torch
from sklearn.linear_model import Ridge
import train_task3 as T
from diagnose import BIN, SPACES, load_clean

t0, n, series = load_clean("synthetic/sensor.csv")
DAY = T.DAY; tr_end = 7 * DAY; va_end = 10 * DAY
norm = T.read_norm("diagnosis/norm_stats.csv")
norm_std = np.array([[norm[(sp, it)][1] for it in T.ITEMS] for sp in SPACES], dtype=np.float32)
sched = T.load_schedule("synthetic/schedule.csv", t0, n)
sched_arr = np.stack([sched[sp] for sp in SPACES])
sp_idx = {sp: i for i, sp in enumerate(SPACES)}
pts = []
with open("diagnosis/eval_points.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["item"] != "co2": continue
        t = int((datetime.fromisoformat(r["time"]) - t0).total_seconds() // 60 // BIN)
        pts.append((sp_idx[r["space"]], t))
pts = np.array(sorted(set(pts)))
H = 12  # 60분 = 12칸, co2 60분 출력 index = 3

# 실제 재실 (5분 평균 인원)
ppl = np.zeros((len(SPACES), n))
cnt = np.zeros((len(SPACES), n))
with open("synthetic/truth.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        b = int((datetime.fromisoformat(r["time"]) - t0).total_seconds() // 60 // BIN)
        if 0 <= b < n:
            ppl[sp_idx[r["space"]], b] += float(r["people"]); cnt[sp_idx[r["space"]], b] += 1
ppl = ppl / np.maximum(cnt, 1)
occ = ppl >= 0.5

preds = {}
feats = {}
for name, us in (("sched", True), ("nosched", False)):
    fe, raw = T.build_features(t0, n, series, norm, sched, us)
    feats[us] = (fe, raw)
    m = T.GRUForecaster(fe.shape[2], T.FUTURE if us else 0, 64)
    m.load_state_dict(torch.load(f"diagnosis/gru_{name}.pt")); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), 2048):
            x, fut, _ = T.make_batch(fe, raw, sched_arr, norm_std, pts[i:i+2048], us)
            out.append(m(x, fut).numpy()[:, 3])
    preds["GRU_" + name] = raw[pts[:, 0], pts[:, 1], 0] + np.concatenate(out) * norm_std[pts[:, 0], 0]
fe, raw = feats[True]
now = raw[pts[:, 0], pts[:, 1], 0]
true = raw[pts[:, 0], pts[:, 1] + H, 0]
preds["persist"] = now.copy()

# 선형 회귀 (Ridge): 같은 입력을 펼쳐서
def flat(idx, us):
    fe_, raw_ = feats[us]
    x, fut, y = T.make_batch(fe_, raw_, sched_arr, norm_std, idx, us)
    X = np.concatenate([x.numpy().reshape(len(idx), -1), fut.numpy()], 1)
    return X, y.numpy()[:, 3]
tr = T.sample_points(raw, 0, tr_end)
for us in (True, False):
    X, y = flat(tr, us); ok = ~np.isnan(y)
    r = Ridge(alpha=1.0).fit(X[ok], y[ok])
    Xe, _ = flat(pts, us)
    preds["Ridge_" + ("sched" if us else "nosched")] = now + r.predict(Xe) * norm_std[pts[:, 0], 0]

# 상황 나누기 (앞으로 60분 안)
s, t = pts[:, 0], pts[:, 1]
fw = t[:, None] + np.arange(1, H + 1)
sch_now = sched_arr[s, t] > 0
sch_fut = sched_arr[s[:, None], fw]
start_soon = (~sch_now) & (sch_fut.max(1) > 0)
end_soon = sch_now & (sch_fut.min(1) == 0)
occ_now = occ[s, t]; occ_fut = occ[s[:, None], fw]
enter = (~occ_now) & occ_fut.any(1)
leave = occ_now & (~occ_fut).any(1)
change = enter | leave
noshow = start_soon & (~occ_fut.any(1))
unbooked_enter = enter & (sch_fut.max(1) == 0) & (~sch_now)
ok = ~np.isnan(true)

groups = {
    "전체": np.ones(len(pts), bool),
    "앞 60분 실제 재실 변화 없음(평온)": ~change,
    "앞 60분 안에 사람 들어옴": enter,
    "앞 60분 안에 사람 나감": leave,
    "  └ 일정 시작 예정 + 실제 입실": start_soon & enter,
    "  └ 일정 시작 예정인데 노쇼": noshow,
    "  └ 일정 없이 입실": unbooked_enter,
}
names = ["persist", "Ridge_nosched", "GRU_nosched", "Ridge_sched", "GRU_sched"]
tot_err = {k: np.abs(preds[k] - true) for k in names}
print(f"{'상황':34}{'비율':>7}" + "".join(f"{k:>14}" for k in names) + f"{'오차몫(GRU_s)':>14}")
for g, msk in groups.items():
    m_ = msk & ok
    share = tot_err["GRU_sched"][m_].sum() / tot_err["GRU_sched"][ok].sum()
    print(f"{g:34}{m_.sum()/ok.sum()*100:6.1f}%" + "".join(f"{tot_err[k][m_].mean():14.1f}" for k in names) + f"{share*100:13.0f}%")

print("\n공간별 (60분 CO2)")
for si, sp in enumerate(SPACES):
    for g in ("앞 60분 실제 재실 변화 없음(평온)", "앞 60분 안에 사람 들어옴", "  └ 일정 시작 예정인데 노쇼", "  └ 일정 없이 입실"):
        m_ = groups[g] & ok & (s == si)
        if m_.sum() < 5: continue
        print(f"{sp:5}{g:34}{m_.sum():6}" + "".join(f"{tot_err[k][m_].mean():10.1f}" for k in names))

# 일정 효과(GRU_nosched - GRU_sched 오차 감소)가 어디서 오는지
gain = tot_err["GRU_nosched"] - tot_err["GRU_sched"]
print("\n일정 효과(오차 감소 합)의 출처")
for g in ("앞 60분 실제 재실 변화 없음(평온)", "앞 60분 안에 사람 들어옴", "앞 60분 안에 사람 나감", "  └ 일정 시작 예정인데 노쇼", "  └ 일정 없이 입실"):
    m_ = groups[g] & ok
    print(f"{g:34}{gain[m_].sum()/gain[ok].sum()*100:6.0f}%  (평균 {gain[m_].mean():.1f}ppm)")

print("\n=== 상태(국면)별 세분 ===")
pw = t[:, None] + np.arange(-24, 1)   # 지난 2시간
occ_past = occ[s[:, None], pw].any(1)
phase = np.full(len(pts), "", object)
phase[(~occ_now) & (~occ_fut.any(1)) & (~occ_past)] = "A 빈 방이 계속(바닥)"
phase[(~occ_now) & (~occ_fut.any(1)) & occ_past] = "B 퇴실 뒤 빠지는 중"
phase[occ_now & occ_fut.all(1)] = "C 사람 계속 있음(쌓이는 중)"
phase[enter] = "D 60분 안 입실"
phase[leave & ~enter] = "E 60분 안 퇴실"
phase[noshow] = "F 일정 있는데 노쇼"
for p in sorted(set(phase)):
    m_ = (phase == p) & ok
    if not p: p = "(기타)"
    share = tot_err["GRU_sched"][m_].sum() / tot_err["GRU_sched"][ok].sum()
    gshare = gain[m_].sum() / gain[ok].sum()
    print(f"{p:28}{m_.sum()/ok.sum()*100:6.1f}%" + "".join(f"{tot_err[k][m_].mean():10.1f}" for k in names) + f"  오차몫{share*100:4.0f}%  일정효과몫{gshare*100:4.0f}%")
print("열:", names)
# 공간별 노쇼 비중
for si, sp in enumerate(SPACES):
    m_ = ok & (s == si)
    nm = m_ & noshow
    print(sp, "노쇼 칸", nm.sum(), "오차몫", round(tot_err["GRU_sched"][nm].sum()/tot_err["GRU_sched"][m_].sum()*100), "%",
          "노쇼 빼면 MAE", round(tot_err["GRU_sched"][m_ & ~noshow].mean(),1), "전체", round(tot_err["GRU_sched"][m_].mean(),1))

print("\n=== 공간별 국면 비율 / GRU_sched MAE ===")
P = sorted(p for p in set(phase) if p)
print(f"{'':6}" + "".join(f"{p[:6]:>14}" for p in P))
for si, sp in enumerate(SPACES):
    m_ = ok & (s == si)
    print(f"{sp:6}" + "".join(f"{((phase==p)&m_).sum()/m_.sum()*100:6.1f}%/{tot_err['GRU_sched'][(phase==p)&m_].mean() if ((phase==p)&m_).sum() else 0:6.1f}" for p in P))
