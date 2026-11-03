"""
과제 2-4 데이터 진단 (정제한 sensor.csv + schedule.csv만 사용)

  (가) 평일 시간대별 평균 패턴: 수업·점심·퇴실이 보이는가
  (나) 일정 변화 뒤 CO2가 몇 분 만에 반응하는가
  (다) 일정과 실제 CO2 변화가 어긋나는 날은 언제인가
  (라) (다)를 정답 파일(truth.csv)로 채점  ← 채점에만 쓰고 분석에는 안 씀

정제 순서 (clean_stats.py와 같은 규칙)
  중복 제거 → 불가능한 값 제거 → 5분 평균 → 10분 이하 빈칸만 직선 보간 (긴 결측은 비워 둠)

사용:  python3 diagnose.py                       # synthetic/ 를 읽고 diagnosis/ 에 결과 저장
       python3 diagnose.py --dir synthetic --out diagnosis
그래프를 그리려면 matplotlib 필요:  pip3 install matplotlib  (없으면 표와 CSV만 만듦)
"""
import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean

VALID = {"co2": (300, 10000), "temperature": (-10, 50), "humidity": (0.5, 100)}
BIN = 5            # 분석 간격(분)
MAX_FILL = 2       # 이 칸 수(=10분) 이하의 빈칸만 보간
SPACES = ["강의실1", "강의실2", "회의실", "라운지"]


# ============================================================ 1. 불러오기 + 정제
def load_clean(path):
    """반환: t0(첫날 0시), n(칸 수), series[(공간,항목)] = [값 또는 None, ...] (5분 칸)"""
    seen = set()
    sums = defaultdict(lambda: defaultdict(float))
    cnts = defaultdict(lambda: defaultdict(int))
    t0 = t_last = None
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["time"], r["space"], r["item"], r["value"])
        if key in seen:                       # 중복 제거
            continue
        seen.add(key)
        lo, hi = VALID[r["item"]]
        v = float(r["value"])
        if not lo <= v <= hi:                 # 불가능한 값 제거
            continue
        t = datetime.fromisoformat(r["time"])
        rows.append((t, r["space"], r["item"], v))
        t0 = t if t0 is None or t < t0 else t0
        t_last = t if t_last is None or t > t_last else t_last
    t0 = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    for t, sp, item, v in rows:               # 5분 평균
        b = int((t - t0).total_seconds() // (BIN * 60))
        sums[(sp, item)][b] += v
        cnts[(sp, item)][b] += 1
    n = int((t_last - t0).total_seconds() // (BIN * 60)) + 1

    series = {}
    for k in sums:
        s = [sums[k][b] / cnts[k][b] if cnts[k][b] else None for b in range(n)]
        i = 0
        while i < n:                          # 짧은 빈칸만 직선 보간
            if s[i] is None:
                j = i
                while j < n and s[j] is None:
                    j += 1
                if 0 < i and j < n and j - i <= MAX_FILL:
                    a, c = s[i - 1], s[j]
                    for m in range(i, j):
                        s[m] = a + (c - a) * (m - i + 1) / (j - i + 1)
                i = j
            else:
                i += 1
        series[k] = s
    return t0, n, series


def load_schedule(path):
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        r["start"] = datetime.fromisoformat(r["start"])
        r["end"] = datetime.fromisoformat(r["end"])
        out.append(r)
    return out


def bin_of(t, t0):
    return int((t - t0).total_seconds() // (BIN * 60))


def avg(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


# ============================================================ (가) 시간대별 패턴
def hourly_pattern(t0, n, series):
    """평일 시각(0~23시)별 평균. 반환: pat[(공간,항목)][시] = 평균"""
    pat = {}
    for k, s in series.items():
        by_hour = defaultdict(list)
        for b, v in enumerate(s):
            t = t0 + timedelta(minutes=b * BIN)
            if v is not None and t.weekday() < 5:
                by_hour[t.hour].append(v)
        pat[k] = {h: mean(by_hour[h]) for h in range(24) if by_hour[h]}
    return pat


# ============================================================ (나) 일정 변화 뒤 반응
def build_events(sched):
    """수업 시작 / 점심(오전 수업 끝) / 퇴실(그날 마지막 수업 끝) / 회의 시작"""
    ev = defaultdict(list)
    last_end = {}
    for e in sched:
        if e["type"] == "class":
            ev["수업 시작"].append((e["space"], e["start"]))
            if e["end"].hour == 12:                        # 12:00 또는 12:30에 끝나는 오전 수업
                ev["점심"].append((e["space"], e["end"]))
            key = (e["space"], e["end"].date())
            if key not in last_end or e["end"] > last_end[key]:
                last_end[key] = e["end"]
        elif e["type"] == "meeting":
            ev["회의 시작"].append((e["space"], e["start"]))
    ev["퇴실"] = [(sp, t) for (sp, _), t in last_end.items()]
    return ev


def response(t0, series, events, before=6, after=12):
    """이벤트 시각을 0으로 맞춘 CO2 변화량 평균 곡선(-30~+60분)과 반응 시간"""
    result = {}
    for name, evs in events.items():
        curves = []
        for sp, t in evs:
            s = series[(sp, "co2")]
            b0 = bin_of(t, t0)
            if b0 - before < 0 or b0 + after >= len(s):
                continue
            win = s[b0 - before: b0 + after + 1]
            base = avg(win[:3])                            # 일정 30~15분 전 평균 (사람이 미리 오는 영향 제외)
            if base is None or any(v is None for v in win):
                continue                                   # 긴 결측이 걸린 이벤트는 제외
            curves.append([v - base for v in win])
        if not curves:
            continue
        curve = [mean(c[i] for c in curves) for i in range(before + after + 1)]
        final = curve[-1]
        sign = 1 if final >= 0 else -1
        # 반응 시작: 변화량이 50ppm을 처음 넘은 시점 / 절반 도달: 60분 뒤 변화량의 절반을 넘은 시점
        # (음수면 일정 시각보다 먼저 반응 = 사람이 미리 오거나 미리 나감)
        mins = [(i - before) * BIN for i in range(len(curve))]
        start = next((m for m, v in zip(mins, curve) if sign * v >= 50), None)
        half = next((m for m, v in zip(mins, curve) if sign * v >= abs(final) / 2), None)
        result[name] = dict(curve=curve, n=len(curves), start=start, half=half, final=final)
    return result


# ============================================================ (다) 일정과 어긋난 날
def mismatch(t0, series, sched):
    """일정은 있는데 CO2가 거의 안 오른 경우, 일정이 없는데 회의실 CO2가 오른 경우"""
    flagged, checked = [], []
    for e in sched:
        if e["type"] not in ("class", "meeting"):
            continue
        s = series[(e["space"], "co2")]
        b0, b1 = bin_of(e["start"], t0), bin_of(e["end"], t0)
        pre = avg(s[max(0, b0 - 6): b0])                   # 시작 전 30분
        during = [v for v in s[b0 + 3: b1] if v is not None]  # 시작 15분 뒤부터 끝까지
        if pre is None or not during:
            continue
        rise = max(during) - pre
        limit = 150 if e["type"] == "class" else 100       # 수업은 사람이 많아 더 크게 올라야 정상
        # 직전 사용으로 CO2가 이미 높으면(600ppm 이상) 올라갈 여지가 작아 '안 올랐다'로 판단할 수 없음
        hold = pre >= 600
        row = dict(space=e["space"], type=e["type"], title=e["title"], start=e["start"],
                   end=e["end"], pre=round(pre), rise=round(rise), hold=hold,
                   flag=(rise < limit) and not hold)
        checked.append(row)
        if row["flag"] or (hold and rise < limit):
            flagged.append(row)

    # 예약 없는 회의실 사용: 30분 사이 CO2가 100ppm 넘게 오른 구간 (예약 앞뒤 30분은 제외)
    s = series[("회의실", "co2")]
    booked = set()
    for e in sched:
        if e["space"] == "회의실":
            booked.update(range(bin_of(e["start"], t0) - 6, bin_of(e["end"], t0) + 6))
    unbooked, b = [], 6
    while b < len(s):
        if b not in booked and s[b] is not None and s[b - 6] is not None and s[b] - s[b - 6] > 100:
            unbooked.append(dict(space="회의실", start=t0 + timedelta(minutes=(b - 6) * BIN),
                                 end=t0 + timedelta(minutes=b * BIN), rise=round(s[b] - s[b - 6])))
            b += 12                                        # 같은 사용을 두 번 세지 않게 1시간 건너뜀
        else:
            b += 1
    return checked, flagged, unbooked


# ============================================================ (라) 정답으로 채점
def score(truth_path, checked, unbooked):
    people = defaultdict(dict)
    for r in csv.DictReader(open(truth_path, encoding="utf-8")):
        people[r["space"]][r["time"][:16]] = int(r["people"])

    def actual(sp, a, b):
        t, vals = a, []
        while t < b:
            v = people[sp].get(t.isoformat()[:16])
            if v is not None:
                vals.append(v)
            t += timedelta(minutes=1)
        return mean(vals) if vals else 0

    tp = fp = fn = 0
    for r in checked:
        empty = actual(r["space"], r["start"], r["end"]) < 0.5     # 실제로 아무도 안 옴
        r["truth"] = "실제 비었음" if empty else "실제 사용"
        tp += r["flag"] and empty
        fp += r["flag"] and not empty
        fn += (not r["flag"]) and empty
    for r in unbooked:   # 상승한 30분 동안 한 번이라도 사람이 있었는지
        t, used = r["start"], False
        while t <= r["end"]:
            used = used or people["회의실"].get(t.isoformat()[:16], 0) > 0
            t += timedelta(minutes=1)
        r["truth"] = "실제 사용" if used else "사용 아님"
    return tp, fp, fn


# ============================================================ 그래프
def plot(out, pat, resp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib이 없어 그래프는 건너뜀. 설치: pip3 install matplotlib)")
        return
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if name in have:                                   # 한글이 나오는 글꼴 고르기
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for sp in SPACES:
        for i, item in enumerate(("co2", "temperature")):
            p = pat.get((sp, item), {})
            hs = sorted(p)
            ax[i].plot(hs, [p[h] for h in hs], marker="o", ms=3, label=sp)
    ax[0].set_title("평일 시간대별 평균 CO2 (정제 후)")
    ax[0].set_ylabel("ppm")
    ax[1].set_title("평일 시간대별 평균 온도 (정제 후)")
    ax[1].set_ylabel("℃")
    for a in ax:
        a.set_xlabel("시")
        a.set_xticks(range(0, 24, 2))
        a.axvspan(12, 13, color="gray", alpha=0.12)
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hourly_pattern.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [(i - 6) * BIN for i in range(19)]
    for name, r in resp.items():
        ax.plot(x, r["curve"], marker="o", ms=3, label=f"{name} (n={r['n']})")
    ax.axvline(0, color="black", lw=1)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("일정 시각 기준 (분)")
    ax.set_ylabel("CO2 변화량 (ppm, 일정 30~15분 전 대비)")
    ax.set_title("일정 변화 뒤 CO2 반응 (평균)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "response_curve.png"), dpi=130)
    print(f"  그래프 저장: {out}/hourly_pattern.png, {out}/response_curve.png")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic")
    ap.add_argument("--out", default="diagnosis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0, n, series = load_clean(os.path.join(a.dir, "sensor.csv"))
    sched = load_schedule(os.path.join(a.dir, "schedule.csv"))

    # (가)
    pat = hourly_pattern(t0, n, series)
    print("(가) 평일 시간대별 평균 CO2 (ppm, 정제 후)")
    hours = range(6, 22)
    print(f"  {'':6}" + "".join(f"{h:>6}" for h in hours))
    for sp in SPACES:
        p = pat[(sp, "co2")]
        print(f"  {sp:6}" + "".join(f"{p.get(h, 0):6.0f}" for h in hours))
    with open(os.path.join(a.out, "hourly_pattern.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["space", "item"] + list(range(24)))
        for (sp, item), p in sorted(pat.items()):
            w.writerow([sp, item] + [round(p[h], 2) if h in p else "" for h in range(24)])

    # (나)
    resp = response(t0, series, build_events(sched))
    print()
    print("(나) 일정 변화 뒤 CO2 반응 (일정 30~15분 전 평균 대비, 음수 분 = 일정보다 먼저)")
    for name, r in resp.items():
        print(f"  {name:6} 이벤트 {r['n']:3}개  60분 뒤 변화 {r['final']:+6.0f}ppm  "
              f"50ppm 변화까지 {r['start']}분  변화의 절반까지 {r['half']}분")
    with open(os.path.join(a.out, "response_curve.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minutes"] + list(resp))
        for i in range(19):
            w.writerow([(i - 6) * BIN] + [round(r["curve"][i], 1) for r in resp.values()])

    # (다)
    checked, flagged, unbooked = mismatch(t0, series, sched)
    tp, fp, fn = score(os.path.join(a.dir, "truth.csv"), checked, unbooked)
    print()
    print(f"(다) 일정은 있는데 CO2가 거의 안 오른 경우 (일정 {len(checked)}건 중 {len(flagged)}건)")
    for r in flagged:
        tag = "판단 보류(직전 사용으로 이미 높음)" if r["hold"] else "의심"
        print(f"  {r['start']:%m-%d(%a) %H:%M}~{r['end']:%H:%M} {r['space']:5} {r['title']:10} "
              f"시작 전 {r['pre']:5}ppm 상승 {r['rise']:+5}ppm  {tag:20} [정답: {r['truth']}]")
    print(f"    예약이 없는데 회의실 CO2가 오른 경우 {len(unbooked)}건")
    for r in unbooked:
        print(f"  {r['start']:%m-%d(%a) %H:%M}  30분 사이 +{r['rise']}ppm   [정답: {r['truth']}]")
    with open(os.path.join(a.out, "mismatch.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["space", "type", "title", "start", "end", "co2_before", "co2_rise",
                    "suspect", "hold", "truth"])
        for r in checked:
            w.writerow([r["space"], r["type"], r["title"], r["start"].isoformat(),
                        r["end"].isoformat(), r["pre"], r["rise"], int(r["flag"]),
                        int(r["hold"]), r["truth"]])

    # (라)
    print()
    print("(라) 채점 (truth.csv): '의심'으로 고른 일정이 실제로 비어 있었나 (판단 보류는 제외)")
    print(f"  맞게 찾음 {tp}건, 잘못 찾음(실제는 사용) {fp}건, 놓침 {fn}건")
    print()
    plot(a.out, pat, resp)
    print(f"  표 저장: {a.out}/hourly_pattern.csv, response_curve.csv, mismatch.csv")


if __name__ == "__main__":
    main()
