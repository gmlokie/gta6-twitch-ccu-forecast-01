# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/gmlokie/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/gmlokie/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour Twitch CCU forecaster for GTA VI launch day: per-hour log-ratio regression on hour-1 viewers over 38 comparable launches; practice loss 0.169 vs house baseline 0.387.

## What it does

`main.py` (Python 3.13, stdlib only, no network) reads one case file (`/input/<id>.in`, schema `makefun/twitch-ccu-case/v1`) and writes `/output/<id>.out`: for each of hours 2–24, five quantiles of that hour's Twitch concurrent viewers in the Grand Theft Auto VI category.

## Method

The launch-day curve is modelled as ratios to the hour-1 print, fitted at run time from the case file's own comparables — never hard-coded:

- **Training set**: comparable launches with a solid release day (hour-1 print present, ≥20 hourly points, hour-1 viewers ≥10,000) and tier "ok" — early-access-contaminated launches and weak analogues (free-to-play, slow-burn) are excluded via the unlock note.
- **Per-hour regression**: for each hour 2–24, regress log(viewers_h / viewers_1) on log(viewers_1) across the comparables. Big hour-1 launches are already near their peak (peak-ratio slope ≈ −0.9 on log hour-1); small hour-1 launches ramp hard once the US unlocks roll in (05:00Z East, 08:00Z West).
- **Winsorised prediction**: evaluated at this case's hour-1 print, clamped to the comparables' observed range, so an unprecedented GTA VI hour-1 cannot extrapolate the curve into a collapse.
- **Spread**: quantiles are normal with sd = 0.85 × the per-hour residual sd (floor 0.20 log units); predicted log-ratios and sds are smoothed over neighbouring hours.
- **Fallbacks**: relax the tier filter, then the hour-1 floor, then the median ratio curve, then the house persistence baseline — so the program always emits a valid forecast.

## Local grade

Practice answers reconstructed from SullyGnome's hourly averages (the kit's own source; hour-1 matches each case file exactly), scored with the rules' pinball loss on the log scale:

- **Overall loss 0.169** (per case: Wolverine 0.209, RDR2 0.153, God of War Ragnarök 0.129, Spider-Man 2 0.186)
- House baseline on the same answers: **0.387** — typical miss 29% vs 83%, 90% coverage 0.91 vs 0.78
- Deterministic: two runs reproduce byte for byte; ~0.2 s for all four practice cases

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
