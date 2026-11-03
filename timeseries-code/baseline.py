"""
과제 2-3, 2-5 기간 나누기 · 정규화 · 기준선 성능 (정제한 sensor.csv만 사용)

  ① 학습·검증·시험 기간을 시간 순서대로 나눔 (기본 7일 / 3일 / 나머지)
  ② 정규화 값(평균·표준편차)을 학습 기간으로만 계산 → norm_stats.csv
  ③ 평가 시점 목록을 만들어 저장 → eval_points.csv  (과제 3 모델도 이 목록으로 평가)
  ④ 기준선 3가지의 5·15·30·60분 뒤 CO2·온도 MAE → baseline_results.csv

사용:  python3 baseline.py                         # synthetic/ 를 읽고 diagnosis/ 에 저장
       python3 baseline.py --train-days 7 --val-days 3
정제 규칙은 diagnose.py 와 같음 (같은 폴더에 diagnose.py 가 있어야 함)
"""
import argparse
import csv
import os
from datetime import datetime, timedelta
from statistics import mean, pstdev

from diagnose import BIN, SPACES, load_clean
# BIN: 데이터를 몇 분 단위로 묶음 (5분)
# SPACES: 측정한 공간 (강의실1, 강의실2, 회의실, 라운지)
# load_clean: 원본 데이터를 읽어서 정리(결측치나 이상치 처리 등)한 뒤 돌려주는 함수

ITEMS = ("co2", "temperature")
HORIZONS = (5, 15, 30, 60)              # 몇 분 뒤
DAY = 24 * 60 // BIN                    # 하루 = 288칸
HOUR = 60 // BIN                        # 1시간 = 12칸
BASELINES = ("현재값 유지", "최근 1시간 평균", "어제 같은 시각")


def predict(s, t, h, name):
    """시각 t(칸)에서 h칸 뒤 값을 기준선 name으로 예측. 계산할 수 없으면 None"""
    # s: 5분 간격 값 (한 공간·한 항목의 값 목록)
    # t: 예측하는 현재 시점 (칸 번호)
    # h: 몇 칸 뒤를 예측 (1칸 = 5분, 12칸 = 60분)
    # name: 사용할 기준선 이름
    if name == "현재값 유지":
        return s[t]
    if name == "최근 1시간 평균":
        w = [v for v in s[t - HOUR + 1: t + 1] if v is not None]
        return mean(w) if len(w) >= HOUR * 3 // 4 else None   # 1시간 중 3/4 이상 있어야 계산
    if name == "어제 같은 시각":
        i = t + h - DAY
        return s[i] if i >= 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic")
    ap.add_argument("--out", default="diagnosis")
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--val-days", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)   # 결과 폴더 만들기 (이미 있으면 그대로 둠)

    t0, n, series = load_clean(os.path.join(a.dir, "sensor.csv"))
    # 기간 나누는 경계선 (칸 번호)
    tr_end = a.train_days * DAY                    # 학습: [0, tr_end)
    va_end = (a.train_days + a.val_days) * DAY     # 검증: [tr_end, va_end), 시험: [va_end, n)
    day = lambda b: (t0 + timedelta(minutes=b * BIN)).date()

    print("① 기간 나누기 (시간 순서)")
    print(f"  학습 {day(0)} ~ {day(tr_end - 1)}  ({tr_end} 칸)")
    print(f"  검증 {day(tr_end)} ~ {day(va_end - 1)}  ({va_end - tr_end} 칸)")
    print(f"  시험 {day(va_end)} ~ {day(n - 1)}  ({n - va_end} 칸)")

    # ② 정규화 값: 학습 기간으로만
    print()
    print("② 정규화 값 (학습 기간으로만 계산)  ※ 괄호 안은 전체 기간으로 잘못 계산했을 때")
    with open(os.path.join(a.out, "norm_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["space", "item", "mean", "std", "train_start", "train_end"])
        # 모든 공간에 대해 CO2, 온도, 습도까지 세 항목을 처리
        for sp in SPACES:
            for item in ("co2", "temperature", "humidity"):
                # 학습 데이터 vs 전체 데이터
                tr = [v for v in series[(sp, item)][:tr_end] if v is not None]   # 학습 기간 값만 (정규화에 사용)
                al = [v for v in series[(sp, item)] if v is not None]            # 전체 기간 값 (비교 출력용)
                m, sd = mean(tr), pstdev(tr)
                w.writerow([sp, item, round(m, 3), round(sd, 3), day(0), day(tr_end - 1)])
                print(f"  {sp:5} {item:12} 평균 {m:8.1f} ({mean(al):8.1f})   "
                      f"표준편차 {sd:7.1f} ({pstdev(al):7.1f})")

    # ③ 평가 시점: 시험 기간 안에서 세 기준선과 정답이 모두 있는 시점만
    points = []
    for sp in SPACES:
        for item in ITEMS:
            s = series[(sp, item)]
            for t in range(va_end, n):
                ok = True                  # 문제가 하나라도 발견되면 False로 바꿈
                for hm in HORIZONS:
                    h = hm // BIN          # 분을 칸 수로 바꿈 (예: 60분 → 12칸)
                    # t + h >= n: 예측하려는 미래가 데이터 끝을 넘어가면 정답이 없음
                    # s[t + h] is None: 정답 칸이 비어 있음 (긴 결측)
                    if t + h >= n or s[t + h] is None:
                        ok = False
                        break
                    if any(predict(s, t, h, b) is None for b in BASELINES):
                        ok = False
                        break
                if ok:
                    points.append((sp, item, t))
    with open(os.path.join(a.out, "eval_points.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["space", "item", "time"])
        for sp, item, t in points:
            w.writerow([sp, item, (t0 + timedelta(minutes=t * BIN)).isoformat()])
    possible = len(SPACES) * len(ITEMS) * (n - va_end - max(HORIZONS) // BIN)
    print()
    print(f"③ 평가 시점 {len(points)}개 저장 (시험 기간 가능한 시점의 {len(points) / possible * 100:.1f}%)")
    print("   제외: 현재값·정답·최근 1시간·어제 값 중 하나라도 비어 있는 시점")

    # ④ 기준선 MAE
    err = {}
    for sp, item, t in points:
        s = series[(sp, item)]
        for hm in HORIZONS:
            h = hm // BIN
            for b in BASELINES:
                # 평가 시점 × 예측 시점(4개) × 기준선(3개)마다 예측값과 정답의 차이
                e = abs(predict(s, t, h, b) - s[t + h])
                err.setdefault((item, hm, b), []).append(e)
                err.setdefault((sp, item, hm, b), []).append(e)

    print()
    print("④ 기준선 성능표: 시험 기간 MAE (같은 평가 시점)")
    with open(os.path.join(a.out, "baseline_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["space", "item", "horizon_min", "baseline", "mae", "n"])
        for item, unit in (("co2", "ppm"), ("temperature", "℃")):
            print(f"  [{item} ({unit})]      " + "".join(f"{f'{hm}분 뒤':>10}" for hm in HORIZONS))
            for b in BASELINES:
                vals = [mean(err[(item, hm, b)]) for hm in HORIZONS]
                fmt = "{:10.1f}" if item == "co2" else "{:10.2f}"
                print(f"  {b:14}" + "".join(fmt.format(v) for v in vals))
                for hm, v in zip(HORIZONS, vals):
                    w.writerow(["전체", item, hm, b, round(v, 3), len(err[(item, hm, b)])])
            print()
        # 공간별 표는 "60분 뒤 CO2 예측 오차(MAE)" 하나만 골라서 보여 줌 (나머지는 CSV에 저장)
        print("  [공간별 60분 뒤 CO2 MAE (ppm)]")
        print(f"  {'':14}" + "".join(f"{sp:>8}" for sp in SPACES))   # 표 머리줄: 공간 이름
        for b in BASELINES:
            row = []
            for sp in SPACES:
                v = mean(err[(sp, "co2", 60, b)])
                row.append(v)
                for item in ITEMS:
                    for hm in HORIZONS:
                        e = err[(sp, item, hm, b)]
                        w.writerow([sp, item, hm, b, round(mean(e), 3), len(e)])
            print(f"  {b:14}" + "".join(f"{v:8.1f}" for v in row))
    print()
    print(f"  저장: {a.out}/norm_stats.csv, eval_points.csv, baseline_results.csv")


if __name__ == "__main__":
    main()
