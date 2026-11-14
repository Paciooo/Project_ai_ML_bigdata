"""Render the deck's target definitions as LaTeX-style equation images.

matplotlib mathtext (Computer Modern), transparent PNG at 600 dpi; the
physical size of each image matches the point size, and presentation/eq/
equations.json stores width/height in inches for build_deck.js.

Usage: python tools/render_equations.py [presentation/eq]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"
INK = "#1F1F1F"
DPI = 600

matplotlib.rcParams["font.family"] = "Arial"
# (text, point size, offset of the line top from the image top in inches)
EQUATIONS = {
    "ret": [(r"$r_{t+1} = \ln\left(\dfrac{C_{t+1}}{C_{t}}\right)$", 13, 0.0)],
    "dir": [(r"$y_{t+1} = \mathrm{UP} \quad \mathrm{if}\ \ C_{t+1} > O_{t+1}$", 12, 0.0),
            (r"doji ($C_{t+1} = O_{t+1}$) = miss", 9.5, 0.27)],
    "vol": [(r"$\mathrm{RV}_{t+1} = \sum_{i \in t+1} r_{i}^{2}$", 13, 0.0),
            (r"$r_{i}$: 1-min log returns in candle $t+1$", 9, 0.47)],
}


def render(lines, path):
    fig = plt.figure(figsize=(6, 2))
    for text, size, dy in lines:
        fig.text(0.0, 1.0 - dy / 2, text, fontsize=size, color=INK, va="top", ha="left")
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    return {"w": round(w / DPI, 4), "h": round(h / DPI, 4)}


def main(out="presentation/eq"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: {"file": f"{k}.png", **render(v, out / f"{k}.png")} for k, v in EQUATIONS.items()}
    (out / "equations.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main(*sys.argv[1:])
