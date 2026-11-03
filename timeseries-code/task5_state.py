"""
과제 5 상태 추정: 센서만 보고 (1) 지금 공간을 쓰는 중인지, (2) 30분 안에 CO2 1,000ppm을 넘을지 맞히기

  ① diagnose.load_clean 으로 같은 정제, 과제 3·4와 같은 기간 나누기 (학습 7일 / 검증 3일 / 나머지 시험)
     정규화도 같은 방식 (학습 기간 평균·표준편차)
  ② 사용 여부 분류
     - 정답(학습·검증용): 시간표·예약에서 만듦 (그 5분 칸에 일정이 있으면 '사용중')
     - 그런데 일정은 계획일 뿐 → truth.csv(실제 인원)와 비교해 일정 정답이 얼마나 틀리는지 먼저 확인
     - 모델 입력에는 일정을 넣지 않음 (센서 + 시각만)
     - 시험 채점은 truth.csv(실제 인원) 기준. 비교 규칙: "최근 15분 CO2가 30ppm 이상 오르면 사용중"
  ③ 환기 선제 경보
     - 양성: 지금 1,000ppm 미만인데 30분 안에 1,000ppm 이상이 됨. 이미 넘은 시점은 평가에서 뺌
     - 양성이 아주 적음(불균형) → 정확도 대신 재현율·macro-F1 로 평가
     - 경보 기준(확률 문턱)은 검증 기간에서 "재현율 85% 이상 중 macro-F1 최고"로 고름
     - 비교 규칙: "지금 850ppm 이상이면 경보"
  ④ 두 분류기의 혼동 행렬, 가장 많이 틀리는 상황(공간·시간대·이유)

사용:  python3 task5_state.py                     # synthetic/ 를 읽고 diagnosis5/ 에 결과 저장
       python3 task5_state.py --dir 공식데이터폴더
같은 폴더에 diagnose.py, train_task3.py 가 있어야 함 (과제 3·4와 같은 함수를 그대로 씀)
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

from diagnose import BIN, SPACES, load_clean
from train_task3 import load_schedule, forward_fill   # 과제 3과 같은 일정 읽기·빈칸 채우기

ITEMS = ("co2", "temperature", "humidity")
DAY = 24 * 60 // BIN            # 하루 = 288칸
HOUR = 60 // BIN                # 1시간 = 12칸
WINDOW = HOUR                   # 모델은 최근 1시간(12칸)을 봄
RISE_LEN = 15 // BIN            # 단순 규칙 ①: 최근 15분(3칸)
RISE_PPM = 30                   #              30ppm 이상 오르면 '사용중'
LIMIT = 1000                    # 환기 기준 CO2
AHEAD = 30 // BIN               # 30분(6칸) 안에 넘는지
RULE_PPM = 850                  # 단순 규칙 ②: 지금 850ppm 이상이면 경보
TARGET_RECALL = 0.85


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
            raw[s, :, i] = [np.nan if v is None else v for v in series.get((sp, item), [None] * n)]
    return raw


def load_people(path, t0, n):
    """truth.csv(1분 간격 실제 인원) → people[공간, 칸] = 5분 평균 인원 (없으면 NaN)"""
    tot = np.zeros((len(SPACES), n))
    cnt = np.zeros((len(SPACES), n))
    sp_idx = {sp: i for i, sp in enumerate(SPACES)}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = sp_idx.get(r["space"])
            b = int((datetime.fromisoformat(r["time"]) - t0).total_seconds() // 60 // BIN)
            if s is None or not 0 <= b < n:
                continue
            tot[s, b] += float(r["people"])
            cnt[s, b] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)


def make_features(raw, t0, n, mean, std):
    """공간마다 (칸, 특성) 배열. 일정 정보는 넣지 않음!
    특성: 센서 3개(공간별 정규화) + CO2 절대값((CO2-1000)/200) + 5분 CO2 변화량 + 결측 표시
          + 시각 sin·cos + 주말 여부
    CO2 절대값을 따로 넣는 이유: 1,000ppm 경보는 '공간 평균보다 얼마나 높냐'가 아니라 '실제 몇 ppm이냐'가 중요해서"""
    times = [t0 + timedelta(minutes=b * BIN) for b in range(n)]
    tod = np.array([(t.hour * 60 + t.minute) / 1440 for t in times], dtype=np.float32)
    weekend = np.array([t.weekday() >= 5 for t in times], dtype=np.float32)
    feats = []
    for s in range(len(SPACES)):
        cols = []
        miss = np.isnan(raw[s, :, 0]).astype(np.float32)
        for i in range(len(ITEMS)):
            cols.append(forward_fill((raw[s, :, i] - mean[s, i]) / std[s, i]))
        co2 = forward_fill(raw[s, :, 0])     # 결측은 바로 앞 값으로
        cols.append((co2 - LIMIT) / 200)
        cols.append(np.diff(co2, prepend=co2[0]) / 50)
        cols += [miss, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), weekend]
        feats.append(np.stack(cols, axis=1))
    return np.stack(feats).astype(np.float32)   # (공간, 칸, 특성 10개)


# ============================================================ 모델 (두 분류기 모두 같은 모양)
class GRUClassifier(nn.Module):
    """최근 1시간을 GRU로 요약 → '예/아니오' 점수 하나"""
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(1)   # 점수(logit). sigmoid 를 거치면 확률


def batch_x(feats, idx):
    s, t = idx[:, 0], idx[:, 1]
    win = t[:, None] + np.arange(-WINDOW + 1, 1)
    return torch.from_numpy(feats[s[:, None], win])


def predict(model, feats, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            out.append(torch.sigmoid(model(batch_x(feats, idx[i:i + 4096]))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(feats, tr_idx, tr_y, va_idx, va_y, a, pos_weight=1.0, name=""):
    """학습 기간으로 배우고, 검증 손실이 가장 낮을 때의 모델을 씀 (시험 기간은 안 봄)"""
    model = GRUClassifier(feats.shape[2], a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))   # 양성이 드물면 양성 오답에 벌점을 더 줌
    yv = torch.from_numpy(va_y.astype(np.float32))
    best, best_state, wait = float("inf"), None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr_idx))
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            loss = lossf(model(batch_x(feats, tr_idx[b])), torch.from_numpy(tr_y[b].astype(np.float32)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(batch_x(feats, va_idx[i:i + 4096])), yv[i:i + 4096]).item()
                                for i in range(0, len(va_idx), 4096)]))
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= a.patience:
            break
    print(f"    {name}: {ep} epoch 학습, 검증 손실 최저 {best:.4f}")
    model.load_state_dict(best_state)
    return model


# ============================================================ 채점 도구
def scores(y, p):
    """정답 y, 예측 p (0/1) → 정확도, 정밀도, 재현율, F1, macro-F1, 혼동 행렬"""
    y, p = y.astype(bool), p.astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # 음성 쪽 F1 (음성을 '맞힐 대상'으로 놓고 같은 계산)
    np_ = tn / (tn + fn) if tn + fn else 0.0
    nr = tn / (tn + fp) if tn + fp else 0.0
    nf1 = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0
    return dict(acc=(tp + tn) / max(1, len(y)), prec=prec, rec=rec, f1=f1, macro=(f1 + nf1) / 2,
                fpr=fp / (fp + tn) if fp + tn else 0.0, tp=tp, fp=fp, fn=fn, tn=tn)


def print_cm(title, m, pos="사용중", neg="비어있음"):
    w = max(len(pos), len(neg)) + 6
    print(f"    [{title}]")
    print(f"      {'':{w}}{'예측 ' + neg:>14}{'예측 ' + pos:>14}")
    print(f"      {'실제 ' + neg:{w}}{m['tn']:>14}{m['fp']:>14}")
    print(f"      {'실제 ' + pos:{w}}{m['fn']:>14}{m['tp']:>14}")


def hour_band(h):
    return "밤(0~7시)" if h < 8 else "오전(8~11시)" if h < 12 else "오후(12~17시)" if h < 18 else "저녁(18~23시)"


def top_errors(rows, key, k=4):
    """틀린 칸 목록에서 key 별 개수 상위 k개 → '이름 n건(비율%)' 문자열"""
    c = Counter(key(r) for r in rows)
    tot = max(1, len(rows))
    return ", ".join(f"{name} {v}건({v / tot * 100:.0f}%)" for name, v in c.most_common(k))


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic")
    ap.add_argument("--out", default="diagnosis5")
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--val-days", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------------- ① 준비
    t0, n, series = load_clean(os.path.join(a.dir, "sensor.csv"))
    tr_end, va_end = a.train_days * DAY, (a.train_days + a.val_days) * DAY
    day = lambda b: (t0 + timedelta(minutes=b * BIN)).date()
    print("① 기간 나누기 (과제 3·4와 같음)")
    print(f"  학습 {day(0)} ~ {day(tr_end - 1)} / 검증 {day(tr_end)} ~ {day(va_end - 1)} / 시험 {day(va_end)} ~ {day(n - 1)}")
    raw = to_array(series, n)
    mean = np.nanmean(raw[:, :tr_end], axis=1)          # 학습 기간으로만 (공간, 항목)
    std = np.nanstd(raw[:, :tr_end], axis=1)
    std[std == 0] = 1.0
    feats = make_features(raw, t0, n, mean, std)
    sched = load_schedule(os.path.join(a.dir, "schedule.csv"), t0, n)
    sched_arr = np.stack([sched[sp] for sp in SPACES])
    people = load_people(os.path.join(a.dir, "truth.csv"), t0, n)
    truth_occ = people >= 0.5                           # 5분 평균 0.5명 이상이면 실제 '사용중'
    hours = np.array([(t0 + timedelta(minutes=b * BIN)).hour for b in range(n)])
    co2 = raw[:, :, 0]

    # 일정이 하나도 없는 공간(예: 라운지)은 일정으로 정답을 만들 수 없음 → 학습에서 뺌
    booked = [s for s in range(len(SPACES)) if sched_arr[s].sum() > 0]
    unbooked = [SPACES[s] for s in range(len(SPACES)) if s not in booked]
    if unbooked:
        print(f"  ※ 일정이 없는 공간 {unbooked}: 일정 정답을 만들 수 없어 사용 여부 학습에서 뺌 (시험 때만 참고로 채점)")

    def points(start, end, spaces, extra=None):
        """[start, end) 안에서 최근 1시간이 데이터 안에 있고, 지금 CO2 값이 있는 (공간, 칸)"""
        out = []
        for s in spaces:
            for t in range(max(start, WINDOW), end):
                if np.isnan(co2[s, t]) or (extra is not None and not extra(s, t)):
                    continue
                out.append((s, t))
        return np.array(out, dtype=np.int64).reshape(-1, 2)

    rows_out = []   # 마지막 요약 표 (results.csv)

    # ---------------------------------------------------------------- ② 사용 여부
    print()
    print("② 사용 여부 분류 (사용중 / 비어있음)")
    has_truth = lambda s, t: not np.isnan(people[s, t])
    # (가) 일정 정답이 실제와 얼마나 다른가 (일정 있는 공간, 전체 기간)
    ck = points(0, n, booked, has_truth)
    sl = sched_arr[ck[:, 0], ck[:, 1]] > 0
    tl = truth_occ[ck[:, 0], ck[:, 1]]
    m = scores(tl, sl)
    print(f"  (가) 일정 정답 vs 실제 인원 (일정 있는 공간, 5분 칸 {len(ck)}개)")
    print(f"      일치 {m['acc'] * 100:.1f}%  /  일정엔 있는데 실제론 빔(노쇼·조기종료) {m['fp']}칸"
          f"  /  일정 없는데 실제론 사용(연장·즉석 사용) {m['fn']}칸")
    for s in booked:
        k = ck[:, 0] == s
        print(f"      {SPACES[s]:6} 일치 {np.mean(sl[k] == tl[k]) * 100:5.1f}%")

    # (나) 학습: 정답 = 일정, 입력 = 센서만
    tr = points(0, tr_end, booked)
    va = points(tr_end, va_end, booked)
    set_seed(a.seed)
    print("  (나) GRU 학습 (정답 = 일정, 입력 = 센서·시각만)")
    occ_model = train(feats, tr, sched_arr[tr[:, 0], tr[:, 1]], va, sched_arr[va[:, 0], va[:, 1]], a,
                      name="사용 여부 모델")

    # (다) 시험: truth 기준 채점
    te = points(va_end, n, range(len(SPACES)), has_truth)
    y_te = truth_occ[te[:, 0], te[:, 1]]
    p_model = predict(occ_model, feats, te) >= 0.5
    rise = np.array([co2[s, t] - co2[s, t - RISE_LEN] for s, t in te])
    p_rule = np.nan_to_num(rise, nan=0) >= RISE_PPM
    p_sched = sched_arr[te[:, 0], te[:, 1]] > 0            # 참고: 일정을 그대로 답으로 쓰면?
    is_b = np.isin(te[:, 0], booked)
    print(f"  (다) 시험 기간 채점 (정답 = 실제 인원, 일정 있는 공간 {is_b.sum()}칸 기준)")
    print(f"      {'':24}{'정확도':>8}{'F1':>8}{'재현율':>8}{'빔→사용중 오류율':>18}")
    occ_res = {}
    for name, p in (("규칙(15분 30ppm 상승)", p_rule), ("GRU(센서만)", p_model), ("참고: 일정 그대로", p_sched)):
        mm = scores(y_te[is_b], p[is_b])
        occ_res[name] = mm
        print(f"      {name:24}{mm['acc'] * 100:7.1f}%{mm['f1']:8.3f}{mm['rec'] * 100:7.1f}%{mm['fpr'] * 100:17.1f}%")
        rows_out.append(["사용 여부", name, round(mm["acc"], 4), round(mm["f1"], 4), round(mm["macro"], 4),
                         round(mm["rec"], 4), round(mm["prec"], 4), round(mm["fpr"], 4),
                         mm["tp"], mm["fp"], mm["fn"], mm["tn"]])
    print("      공간별 F1 (GRU / 규칙)   " + "  ".join(
        f"{SPACES[s]} {scores(y_te[te[:, 0] == s], p_model[te[:, 0] == s])['f1']:.2f}"
        f"/{scores(y_te[te[:, 0] == s], p_rule[te[:, 0] == s])['f1']:.2f}"
        + ("(일정없음·참고)" if s not in booked else "") for s in range(len(SPACES))))
    print_cm("GRU 혼동 행렬 (일정 있는 공간)", occ_res["GRU(센서만)"])

    # 틀린 이유 붙이기 (실제 인원 기록으로 판단)
    def occ_reason(s, t, truth, pred):
        if pred and not truth:
            recent = truth_occ[s, max(0, t - 6):t]
            return "퇴장 직후 30분(CO2 아직 높음)" if recent.any() else "아무도 없는데 CO2 변화(환기·날씨 등)"
        recent = truth_occ[s, max(0, t - RISE_LEN):t]
        if not recent.all():
            return "입장 직후 15분(CO2 아직 안 오름)"
        return "1~3명 소수 인원(CO2가 거의 안 오름)" if people[s, t] <= 3 else "많은 인원인데 CO2가 평평(포화·환기)"
    err_rows = []
    for (s, t), y, p in zip(te, y_te, p_model):
        if y != p and s in booked:
            err_rows.append(dict(task="사용 여부", kind="헛경보(빔→사용중)" if p else "놓침(사용중→빔)",
                                 space=SPACES[s], time=(t0 + timedelta(minutes=int(t) * BIN)).isoformat(),
                                 band=hour_band(hours[t]), reason=occ_reason(s, t, y, p),
                                 co2=round(float(co2[s, t]), 1), people=round(float(people[s, t]), 1)))
    occ_err = err_rows[:]

    # ---------------------------------------------------------------- ③ 환기 선제 경보
    print()
    print(f"③ 환기 선제 경보 (지금 {LIMIT}ppm 미만 → {AHEAD * BIN}분 안에 {LIMIT}ppm 이상이 되면 양성)")

    def future_max(s, t):
        f = co2[s, t + 1:t + 1 + AHEAD]
        return np.nan if np.isnan(f).all() else np.nanmax(f)
    below = lambda s, t: co2[s, t] < LIMIT and t + AHEAD < n and not np.isnan(future_max(s, t))
    all_sp = range(len(SPACES))
    tr2, va2, te2 = points(0, tr_end, all_sp, below), points(tr_end, va_end, all_sp, below), points(va_end, n, all_sp, below)
    lab = lambda idx: np.array([future_max(s, t) >= LIMIT for s, t in idx])
    y_tr2, y_va2, y_te2 = lab(tr2), lab(va2), lab(te2)
    print(f"  양성 비율: 학습 {y_tr2.mean() * 100:.1f}% / 검증 {y_va2.mean() * 100:.1f}% / 시험 {y_te2.mean() * 100:.1f}%"
          "  (이미 넘은 시점은 뺐음)")
    pw = float((~y_tr2).sum() / max(1, y_tr2.sum()))
    set_seed(a.seed)
    alert_model = train(feats, tr2, y_tr2, va2, y_va2, a, pos_weight=pw, name=f"경보 모델 (양성 가중치 {pw:.0f}배)")

    # 문턱 고르기: 검증 기간에서 재현율 85% 이상인 문턱 중 macro-F1 최고
    pv = predict(alert_model, feats, va2)
    best_th, best_m = 0.5, -1
    for th in np.linspace(0.01, 0.99, 99):
        mm = scores(y_va2, pv >= th)
        if mm["rec"] >= TARGET_RECALL and mm["macro"] > best_m:
            best_th, best_m = th, mm["macro"]
    print(f"  문턱: {best_th:.2f} (검증 기간 재현율 {TARGET_RECALL * 100:.0f}% 이상 중 macro-F1 최고 {best_m:.3f})")

    pt = predict(alert_model, feats, te2)
    p_model2 = pt >= best_th
    p_rule2 = co2[te2[:, 0], te2[:, 1]] >= RULE_PPM
    p_never = np.zeros(len(te2), dtype=bool)
    print(f"  시험 기간 채점 ({len(te2)}칸, 양성 {int(y_te2.sum())}칸)")
    print(f"      {'':24}{'정확도':>8}{'재현율':>8}{'정밀도':>8}{'macro-F1':>10}")
    al_res = {}
    for name, p in (("참고: 경보 안 울림", p_never), (f"규칙(지금 {RULE_PPM}ppm 이상)", p_rule2), ("GRU 경보", p_model2)):
        mm = scores(y_te2, p)
        al_res[name] = mm
        print(f"      {name:24}{mm['acc'] * 100:7.1f}%{mm['rec'] * 100:7.1f}%{mm['prec'] * 100:7.1f}%{mm['macro']:10.3f}")
        rows_out.append(["환기 경보", name, round(mm["acc"], 4), round(mm["f1"], 4), round(mm["macro"], 4),
                         round(mm["rec"], 4), round(mm["prec"], 4), round(mm["fpr"], 4),
                         mm["tp"], mm["fp"], mm["fn"], mm["tn"]])
    print_cm("GRU 경보 혼동 행렬", al_res["GRU 경보"], pos="넘을 것", neg="안 넘음")

    def alert_reason(s, t, truth, pred):
        now, fmax = co2[s, t], future_max(s, t)
        if truth and not pred:
            return f"지금 {RULE_PPM}ppm 미만인데 빠르게 오름(갑자기 많이 들어옴)" if now < RULE_PPM else "문턱 근처에서 놓침"
        if fmax >= LIMIT - 100:
            return "900~1,000ppm 근처까지만 오르고 멈춤"
        return "사람이 나가서 CO2가 내려감" if fmax < now else "오르다가 멈춤(인원 적음·환기)"
    for (s, t), y, p in zip(te2, y_te2, p_model2):
        if y != p:
            err_rows.append(dict(task="환기 경보", kind="헛경보" if p else "놓침",
                                 space=SPACES[s], time=(t0 + timedelta(minutes=int(t) * BIN)).isoformat(),
                                 band=hour_band(hours[t]), reason=alert_reason(s, t, y, p),
                                 co2=round(float(co2[s, t]), 1), people=round(float(np.nan_to_num(people[s, t])), 1)))
    al_err = err_rows[len(occ_err):]

    # ---------------------------------------------------------------- ④ 많이 틀리는 상황
    print()
    print("④ GRU가 많이 틀리는 상황 (시험 기간)")
    for title, rows in (("사용 여부", occ_err), ("환기 경보", al_err)):
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r)
        print(f"  [{title}] 틀린 칸 {len(rows)}개")
        for kind, rr in sorted(by_kind.items(), key=lambda x: -len(x[1])):
            print(f"    {kind} {len(rr)}칸")
            print(f"      이유:   {top_errors(rr, lambda r: r['reason'])}")
            print(f"      공간:   {top_errors(rr, lambda r: r['space'])}")
            print(f"      시간대: {top_errors(rr, lambda r: r['band'])}")

    # ---------------------------------------------------------------- 저장
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "accuracy", "f1", "macro_f1", "recall", "precision",
                    "false_alarm_rate", "tp", "fp", "fn", "tn"])
        w.writerows(rows_out)
    with open(os.path.join(a.out, "errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "kind", "space", "time", "band", "reason", "co2", "people"])
        w.writeheader()
        w.writerows(err_rows)
    torch.save(occ_model.state_dict(), os.path.join(a.out, "gru_occupancy.pt"))
    torch.save(alert_model.state_dict(), os.path.join(a.out, "gru_alert.pt"))
    print()
    print(f"  저장: {a.out}/results.csv (요약 표), errors.csv (틀린 칸 하나하나), gru_*.pt")


if __name__ == "__main__":
    main()
