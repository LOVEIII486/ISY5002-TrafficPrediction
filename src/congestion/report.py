"""Figures and tables of the congestion levels for the 8 cameras.

    python -m src.congestion.report
    python -m src.congestion.report --levels results/congestion/congestion_levels.csv --out results/congestion/report

Input: congestion_levels.csv from `ensemble_eval apply` (one level per frame: free / moderate / congested,
from the primary method frozen on the practice labels) and the camera names from the filter config.

Outputs (results/congestion/report/):
- heatmaps_congested.png: per camera, weekday x hour (Singapore time), the share of frames rated congested;
- heatmaps_busy.png: the same for moderate or congested, which shows the daily rhythm of cameras that
  never congest;
- heatmap_<camera>.png: one camera per file (share congested), for slides;
- timeline.png: every frame of the two weeks as a coloured strip per camera (weekends shaded);
- summary.csv / summary.md: per camera the share of each level, weekday / weekend and day / night
  congestion, the peak hour, the most congested weekday-hour slot and the longest congested spell.

Each weekday occurs only twice in the two weeks, so a heatmap cell holds about 12 frames; read the
heatmaps as typical patterns, not as precise rates. Grey hatched cells have no images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LEVELS = ["free", "moderate", "congested"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# sequential blue ramp (0% a faint tint so empty cells still read as cells) and an ordinal 3-step version for the levels
RAMP = ["#e9f0f9", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LEVEL_COLOURS = ["#86b6ef", "#2a78d6", "#104281"]
GAP_MINUTES = 20  # frames further apart than this leave a gap in the timeline / end a congested spell

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(levels_csv: Path, config: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(levels_csv, dtype={"camera_id": str})
    df["time"] = pd.to_datetime(df["captured_at_sgt"])
    df["congested"] = df["level_name"] == "congested"
    df["busy"] = df["level_name"].isin(["moderate", "congested"])
    cfg = json.loads(config.read_text(encoding="utf-8"))
    names = {c: cfg["cameras"].get(c, {}).get("name", c) for c in sorted(df["camera_id"].unique())}
    return df.sort_values(["camera_id", "time"]).reset_index(drop=True), names


def grid(g: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    share = g.pivot_table(index="weekday", columns="hour", values=col, aggfunc="mean").reindex(index=range(7), columns=range(24))
    n = g.pivot_table(index="weekday", columns="hour", values=col, aggfunc="size").reindex(index=range(7), columns=range(24))
    return share.to_numpy(float), n.fillna(0).to_numpy(int)


def draw_heatmap(ax, share: np.ndarray, cmap, title: str) -> object:
    masked = np.ma.masked_invalid(share)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 24, 7, facecolor=SURFACE, hatch="////", edgecolor=GRID, lw=0, zorder=0))
    im = ax.pcolormesh(np.arange(-0.5, 24), np.arange(-0.5, 7), masked, cmap=cmap, vmin=0, vmax=1,
                       edgecolors=SURFACE, linewidth=1.2, zorder=1)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(6.5, -0.5)
    ax.set_yticks(range(7), WEEKDAYS)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}" for h in range(0, 24, 3)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    return im


def heatmaps(df: pd.DataFrame, names: dict, col: str, label: str, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    cams = list(names)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11.5), constrained_layout=True)
    for ax, cam in zip(axes.flat, cams):
        share, _ = grid(df[df["camera_id"] == cam], col)
        im = draw_heatmap(ax, share, cmap, f"{cam} · {names[cam]}")
    for ax in axes.flat[len(cams):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Hour of day (Singapore time)")
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.025, pad=0.02, aspect=50)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_visible(False)
    cb.set_label(f"{label}  ·  hatched grey = no images", color=INK2)
    fig.suptitle(f"{label} by weekday and hour, 13–29 September 2026", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def single_heatmap(g: pd.DataFrame, cam: str, name: str, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    share, _ = grid(g, "congested")
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    im = draw_heatmap(ax, share, cmap, f"{cam} · {name} — share of frames rated congested")
    ax.set_xlabel("Hour of day (Singapore time)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def timeline(df: pd.DataFrame, names: dict, path: Path) -> None:
    cams = list(names)
    start = df["time"].min().normalize()
    end = df["time"].max().normalize() + pd.Timedelta(days=1)
    fig, ax = plt.subplots(figsize=(16, 0.62 * len(cams) + 1.6), constrained_layout=True)
    day = start
    while day < end:  # shade weekends
        if day.weekday() >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1), color="#f0efec", zorder=0, lw=0)
        ax.axvline(day, color=GRID, lw=0.6, zorder=1)
        day += pd.Timedelta(days=1)
    for row, cam in enumerate(cams):
        g = df[df["camera_id"] == cam]
        t = g["time"].to_numpy()
        nxt = np.append(t[1:], t[-1] + np.timedelta64(10, "m"))
        width = np.minimum(nxt - t, np.timedelta64(GAP_MINUTES, "m"))
        lv = g["level"].to_numpy()
        for k, colour in enumerate(LEVEL_COLOURS):
            sel = lv == k
            ax.broken_barh(list(zip(pd.to_datetime(t[sel]), pd.to_timedelta(width[sel]))), (row - 0.36, 0.72),
                           facecolors=colour, lw=0, zorder=2)
    ax.set_yticks(range(len(cams)), [f"{c}  {names[c]}" for c in cams])
    ax.set_ylim(len(cams) - 0.5, -0.5)
    ticks = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    ax.set_xticks(ticks + pd.Timedelta(hours=12), [d.strftime("%a\n%d") for d in ticks])
    ax.set_xlim(start, end)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(LEVEL_COLOURS, ["Free-flowing", "Moderate", "Congested"])] +
              [Patch(facecolor=SURFACE, edgecolor=GRID, label="No image"), Patch(facecolor="#f0efec", label="Weekend")],
              loc="upper left", bbox_to_anchor=(0, -0.12), ncol=5, frameon=False)
    ax.set_title("Congestion level of every frame, 13–29 September 2026 (Singapore time)", loc="left", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def longest_spell(g: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float]:
    best, cur = (None, None, 0.0), None
    prev_t = None
    for t, c in zip(g["time"], g["congested"]):
        if c and cur is not None and prev_t is not None and (t - prev_t) <= pd.Timedelta(minutes=GAP_MINUTES):
            cur = (cur[0], t)
        elif c:
            cur = (t, t)
        else:
            cur = None
        if cur is not None:
            hours = (cur[1] - cur[0]).total_seconds() / 3600 + 10 / 60
            if hours > best[2]:
                best = (cur[0], cur[1], hours)
        prev_t = t
    return best


def summary(df: pd.DataFrame, names: dict, out: Path) -> pd.DataFrame:
    rows = []
    for cam, g in df.groupby("camera_id"):
        share = g["level_name"].value_counts(normalize=True)
        by_hour = g.groupby("hour")["congested"].mean()
        slot = g.groupby(["weekday", "hour"]).agg(mean=("congested", "mean"), size=("congested", "size"),
                                                   level=("level", "mean"))
        slot = slot[slot["size"] >= 6].sort_values(["mean", "level", "size"], ascending=False)  # ties: busier, more frames
        s0, s1, hours = longest_spell(g)
        row = {"camera": cam, "location": names[cam], "frames": len(g),
               "free_%": 100 * share.get("free", 0), "moderate_%": 100 * share.get("moderate", 0),
               "congested_%": 100 * share.get("congested", 0),
               "congested_weekday_%": 100 * g.loc[g["weekday"] < 5, "congested"].mean(),
               "congested_weekend_%": 100 * g.loc[g["weekday"] >= 5, "congested"].mean(),
               "congested_day_%": 100 * g.loc[g["is_day"], "congested"].mean(),
               "congested_night_%": 100 * g.loc[~g["is_day"], "congested"].mean(),
               "slots_mostly_congested": int((slot["mean"] >= 0.5).sum())}
        if g["congested"].any():
            (wd, hr), top = slot.index[0], slot["mean"].iloc[0]
            row.update({"peak_hour": f"{by_hour.idxmax():02d}:00", "peak_hour_congested_%": 100 * by_hour.max(),
                        "peak_slot": f"{WEEKDAYS[wd]} {hr:02d}:00", "peak_slot_congested_%": 100 * top,
                        "longest_spell": f"{s0:%a %d %b %H:%M} – {s1:%H:%M}" if s0.date() == s1.date()
                        else f"{s0:%a %d %b %H:%M} – {s1:%a %d %b %H:%M}", "longest_spell_h": hours})
        else:
            row.update({"peak_hour": "–", "peak_hour_congested_%": 0.0, "peak_slot": "–", "peak_slot_congested_%": 0.0,
                        "longest_spell": "never congested", "longest_spell_h": 0.0})
        rows.append(row)
    table = pd.DataFrame(rows)
    num = table.select_dtypes("number").columns.drop("frames")
    table[num] = table[num].round(1)
    table.to_csv(out / "summary.csv", index=False)
    md = ["| Camera | Location | Frames | Free | Moderate | Congested | Congested weekday / weekend | Congested day / night "
          "| Peak hour | Most congested slot | Weekday-hour slots congested ≥50% of the time (of 168) | Longest congested spell |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.to_dict("records"):
        md.append(f"| {r['camera']} | {r['location']} | {r['frames']:,} | {r['free_%']:.0f}% | {r['moderate_%']:.0f}% | "
                  f"{r['congested_%']:.0f}% | {r['congested_weekday_%']:.0f}% / {r['congested_weekend_%']:.0f}% | "
                  f"{r['congested_day_%']:.0f}% / {r['congested_night_%']:.0f}% | {r['peak_hour']} ({r['peak_hour_congested_%']:.0f}%) | "
                  f"{r['peak_slot']} ({r['peak_slot_congested_%']:.0f}%) | {r['slots_mostly_congested']} | {r['longest_spell']}"
                  + (f" ({r['longest_spell_h']:.1f} h)" if r["longest_spell_h"] else "") + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Congestion heatmaps, timeline and summary table for the 8 cameras")
    ap.add_argument("--levels", type=Path, default=Path("results/congestion/congestion_levels.csv"))
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/congestion/report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df, names = load(args.levels, args.config)

    heatmaps(df, names, "congested", "Share of frames rated congested", args.out / "heatmaps_congested.png")
    heatmaps(df, names, "busy", "Share of frames rated moderate or congested", args.out / "heatmaps_busy.png")
    for cam, g in df.groupby("camera_id"):
        single_heatmap(g, cam, names[cam], args.out / f"heatmap_{cam}.png")
    timeline(df, names, args.out / "timeline.png")
    table = summary(df, names, args.out)
    print(table[["camera", "free_%", "moderate_%", "congested_%", "peak_hour", "peak_slot", "longest_spell", "longest_spell_h"]].to_string(index=False))
    print(f"Wrote heatmaps, timeline and summary -> {args.out}")


if __name__ == "__main__":
    main()
