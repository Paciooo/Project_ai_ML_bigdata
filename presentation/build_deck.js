// Final report deck (NQ only). Every number comes from deck_data.json, written by
//   python report.py results/nq_study/runs/<run id> presentation/deck_data.json
// Build: cd presentation && npm install && node build_deck.js NQ_final_report.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = process.env.DECK_OUT || process.argv[2] || "NQ_final_report.pptx";
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const REPO = "https://github.com/Paciooo/Project_ai_ML_bigdata";

// Waseda Red (RGB 142,23,40), white, gold accent
const RED = "8E1728", RED_DARK = "5E0F1B", TINT = "F6E7E9", GOLD = "B8964B", GREEN = "2E6B3F";
const INK = "1A1A1A", GREY = "7F7F7F", MID = "595959", LINE = "BFBFBF", LIGHT = "E6E6E6", PALE = "F4F4F4";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.author = "Alessandro Pacifici";
pres.title = "Can Machine Learning Trade the Next Nasdaq Candle?";
const W = 13.333, L = 0.55, R = W - 0.55, CW = R - L;
const FOOT = "© Alessandro Pacifici · Waseda University";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { line: { x: L, y: 1.38, w: CW, h: 0, line: { color: INK, width: 1 } } },
    { line: { x: L, y: 6.98, w: CW, h: 0, line: { color: INK, width: 1 } } },
    { text: { text: FOOT, options: { x: R - 4.5, y: 7.03, w: 4.5, h: 0.3, align: "right", fontSize: 9, bold: true, color: RED, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: L, y: 0.32, w: CW, h: 0.55, fontSize: 24, bold: true,
                                color: INK, margin: 0, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: L, y: 0.88, w: CW, h: 0.45, fontSize: 16, color: GREY,
                                margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: { x: L, y: 7.03, w: 0.6, h: 0.3, fontSize: 9, color: GREY, margin: 0 },
});
pres.defineSlideMaster({ title: "COVER", background: { color: RED } });

let sec = null;
function content(title, subtitle, section) {
  if (section !== sec) { pres.addSection({ title: section }); sec = section; }
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addText(subtitle, { placeholder: "subtitle" });
  return s;
}
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: "Arial", color: INK, fontSize: 12,
                                                          margin: 0, valign: "top" }, o));
function header(s, x, y, w, text) {
  T(s, text, { x: x + 0.1, y, w: w - 0.1, h: 0.3, fontSize: 12, bold: true, color: MID, valign: "bottom" });
  s.addShape(pres.shapes.LINE, { x, y: y + 0.36, w, h: 0, line: { color: GREY, width: 1 } });
}
function bullets(s, items, o) {
  const runs = items.map((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    return parts.map((p, j) => ({ text: p.text, options: Object.assign({
      bullet: j === 0 ? { code: "25AA" } : undefined, color: INK,
      breakLine: j === parts.length - 1 && i < items.length - 1, paraSpaceAfter: 7 }, p.o || {}) }));
  }).flat();
  T(s, runs, Object.assign({ fontSize: 12.5 }, o));
}
function card(s, x, y, w, h, num, title, body, color = RED, numW = 0.6) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color, width: 1.25 } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: numW, h, rectRadius: 0.06, fill: { color }, line: { color, width: 1.25 } });
  s.addShape(pres.shapes.RECTANGLE, { x: x + numW - 0.12, y, w: 0.12, h, fill: { color }, line: { type: "none" } });
  T(s, String(num), { x, y, w: numW, h, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  T(s, [{ text: title, options: { bold: true, breakLine: true } }, { text: body, options: { color: "333333" } }],
    { x: x + numW + 0.15, y: y + 0.04, w: w - numW - 0.3, h: h - 0.08, fontSize: 11.5, valign: "middle" });
}
function callout(s, x, y, w, h, text, color = RED, fill = TINT) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color, width: 1, dashType: "dash" } });
  T(s, text, { x: x + 0.15, y, w: w - 0.3, h, fontSize: 12.5, bold: true, color: RED_DARK, align: "center", valign: "middle" });
}
function stat(s, x, y, w, big, label, color = RED) {
  T(s, big, { x, y, w, h: 0.72, fontSize: 32, bold: true, color, valign: "bottom" });
  T(s, label, { x, y: y + 0.78, w, h: 0.62, fontSize: 11.5, color: MID });
}
const hdrCell = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, color: "FFFFFF", fill: { color: RED }, align: "center", valign: "middle" }, o) });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ valign: "middle" }, o) });
const num = (t, o = {}) => cell(t, Object.assign({ align: "center" }, o));
const tableOpts = (x, y, w, colW, extra = {}) => Object.assign({ x, y, w, colW, fontFace: "Arial", fontSize: 11, color: INK,
  border: { type: "solid", pt: 0.75, color: LINE }, margin: [3, 5, 3, 5], rowH: 0.3 }, extra);
const chartText = { catAxisLabelColor: MID, valAxisLabelColor: MID, catAxisLabelFontFace: "Arial", valAxisLabelFontFace: "Arial",
  dataLabelFontFace: "Arial", legendFontFace: "Arial", catAxisLabelFontSize: 10.5, valAxisLabelFontSize: 10, dataLabelFontSize: 9.5,
  dataLabelColor: INK, legendFontSize: 10.5, legendColor: INK };

// ---------------------------------------------------------------- formatting / lookups
const pct = (v, d = 2) => (100 * v).toFixed(d) + "%";
const pp = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d) + " pp";
function usd(v) {
  const a = Math.abs(v), sign = v < 0 ? "−" : "+";
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${sign}$${(a / 1e3).toFixed(1)}k`;
  return `${sign}$${a.toFixed(0)}`;
}
const usd0 = (v) => (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
const FAM = { logreg: "Logistic regression", xgboost: "XGBoost", random_forest: "Random forest", mlp: "Neural network (MLP)" };
const FREQ = { "1min": "1 min", "5min": "5 min", "15min": "15 min", "1h": "1 hour", session: "1 day" };
const FREQS = ["1min", "5min", "15min", "1h", "session"].filter((f) => D.info[f]);
const STRAT = { long_short: "Long & Short", long_only: "Long only", short_only: "Short only" };
function mname(m) {
  const map = {
    always_long: "Always long", always_short: "Always short", reversal: "Reversal rule",
    logreg_original: "Logistic (original)", "logreg_C1_ext": "Logistic C=1 + features", "logreg_C0.01_ext": "Logistic C=0.01 + features",
    xgb_original: "XGBoost (original)", xgb_d3_early_ext: "XGBoost d3, early stop", xgb_d5_early_ext: "XGBoost d5, early stop",
    rf_original: "Random forest (original)", rf_d10_ext: "Random forest d10",
    "mlp_original_64-32-16": "MLP 64-32-16 (original)", "mlp_64-32_ext": "MLP 64-32", "mlp_128-64-32_ext": "MLP 128-64-32",
    "mlp_256-128-64-32_ext": "MLP 256-128-64-32", "mlp_128-64-32_drop0.2_bn_ext": "MLP 128-64-32 + dropout/BN",
    "mlp_256-128-64-32_drop0.2_bn_ext": "MLP 256-128-64-32 + dropout/BN", "mlp_128-64-32_lr3e-4_b1024_ext": "MLP 128-64-32, lr 3e-4",
  };
  return map[m] || m;
}
const acc = (f, m) => D.accuracy_test.find((r) => r.frequency === f && r.model === m);
const grid = (f, m, st, tk = "none") => D.strategies_all_test.find((r) => r.frequency === f && r.model === m && r.strategy === st && r.threshold_kind === tk);
const C = D.choice;                     // pre-specified overall choice (validation)
const F = D.primary_frequency;
const CAND = D.candidates;
const bestTest = [...CAND].sort((a, b) => b.test_net_usd - a.test_net_usd)[0];
const deployable = CAND.filter((c) => c.deployable);
const sig = D.models_all.filter((r) => r.opt_vs_majority_ci[0] > 0);
const bestPred = [...sig].sort((a, b) => b.opt_vs_majority_pp - a.opt_vs_majority_pp)[0];
const bestPredAcc = acc(bestPred.frequency, bestPred.optimised);
const chSplit = (sp) => D.choice_splits.find((r) => r.split === sp);
const chAcc = acc(C.frequency, C.model);

// ================================================================ 1. Cover
{
  pres.addSection({ title: "Report" }); sec = "Report";
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Report" });
  let p = 1.2, seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const yOf = (v) => 6.4 - v;
  for (let i = 0; i < 15; i++) {
    const o = p, c = Math.min(4.6, Math.max(0.5, p + (rnd() - 0.38) * 0.7));
    const hi = Math.max(o, c) + 0.05 + rnd() * 0.25, lo = Math.min(o, c) - 0.05 - rnd() * 0.25;
    const x = 7.7 + i * 0.34, bh = Math.max(Math.abs(c - o), 0.06);
    s.addShape(pres.shapes.LINE, { x: x + 0.1, y: yOf(hi), w: 0, h: hi - lo, line: { color: "FFFFFF", width: 1, transparency: 45 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: yOf(Math.max(o, c)), w: 0.2, h: bh,
      fill: { color: "FFFFFF", transparency: c >= o ? 30 : 85 }, line: { color: "FFFFFF", width: 1, transparency: 35 } });
    p = c;
  }
  T(s, "AI, MACHINE LEARNING AND BIG DATA FOR FINANCE AND MANAGEMENT  ·  FINAL REPORT",
    { x: L + 0.05, y: 0.7, w: 9, h: 0.35, fontSize: 11, bold: true, color: "F1D9A0", charSpacing: 1 });
  T(s, "Can Machine Learning Trade the Next Nasdaq Candle?",
    { x: L + 0.05, y: 1.45, w: 7.0, h: 2.0, fontSize: 38, bold: true, color: "FFFFFF", valign: "bottom" });
  T(s, "Next-candle direction models and long/short backtests on E-mini Nasdaq-100 futures (NQ), after commissions",
    { x: L + 0.05, y: 3.6, w: 6.6, h: 0.9, fontSize: 17, color: "F3DADE" });
  s.addShape(pres.shapes.LINE, { x: L + 0.05, y: 4.85, w: 1.4, h: 0, line: { color: GOLD, width: 2.5 } });
  T(s, [{ text: "Alessandro Pacifici", options: { bold: true, breakLine: true } },
        { text: "Waseda University  ·  November 2026", options: { breakLine: true } },
        { text: "Option 1: data analysis  ·  Code and results: ", options: {} },
        { text: "github.com/Paciooo/Project_ai_ML_bigdata", options: { hyperlink: { url: REPO }, color: "FFFFFF", underline: { style: "sng" } } }],
    { x: L + 0.05, y: 5.05, w: 8, h: 1.2, fontSize: 13, color: "FFFFFF", paraSpaceAfter: 4 });
  s.addNotes("Option 1, data analysis. Question: can machine learning predict the direction of the next Nasdaq futures candle well enough to trade it profitably after commissions?");
}

// ================================================================ 2. Executive summary
{
  const s = content("Executive Summary",
    deployable.length ? "A robust strategy passed every pre-specified test" : "Models predict slightly better than chance, but no strategy survives costs and robustness tests", "Report");
  const sx = [L, L + 3.1, L + 6.2, L + 9.3];
  stat(s, sx[0], 1.55, 2.9, "6.2M", "one-minute NQ bars (2006–2025), tested on 1, 5, 15, 60-minute and daily candles");
  stat(s, sx[1], 1.55, 2.9, pp(bestPred.opt_vs_majority_pp), `best accuracy gain over “always up” (${mname(bestPred.optimised)}, ${FREQ[bestPred.frequency]}, 95% CI ${bestPred.opt_vs_majority_ci[0].toFixed(2)} to ${bestPred.opt_vs_majority_ci[1].toFixed(2)})`);
  stat(s, sx[2], 1.55, 2.9, String(D.n_val_configurations), "model × direction × filter strategies compared on validation data");
  stat(s, sx[3], 1.55, 2.9, String(deployable.length), "strategies passing all tests: profitable out of sample, after costs, and robust", GOLD);
  const cy = 3.3, ch = 3.4, cw = 3.95;
  const cols = [
    ["Problem", "Can a model predict whether the next Nasdaq futures candle closes above its open, and trade that signal profitably after $5 round-trip commissions?"],
    ["Approach", "Original vs optimised logistic regression, XGBoost, random forest and neural networks (up to 4 hidden layers), new causal features, strict chronological splits, long/short, long-only and short-only backtests."],
    ["Verdict", deployable.length ? "See final slide." :
      `Optimisation adds at most a few tenths of a point of accuracy. Unfiltered strategies lose heavily after costs; filtered ones hover around zero. No deployable strategy was found.`],
  ];
  cols.forEach(([h, b], i) => {
    const x = L + i * (cw + 0.19);
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy, w: cw, h: ch, fill: { color: i === 2 ? TINT : PALE }, line: { color: i === 2 ? RED : LIGHT, width: 1 } });
    T(s, h, { x: x + 0.25, y: cy + 0.2, w: cw - 0.5, h: 0.4, fontSize: 16, bold: true, color: RED });
    T(s, b, { x: x + 0.25, y: cy + 0.75, w: cw - 0.5, h: ch - 0.95, fontSize: 14, color: INK });
  });
  s.addNotes(`Key message: machine learning finds a small, statistically detectable directional edge at short horizons, but after commissions none of the ${D.n_val_configurations} validated strategy configurations produces a robust out-of-sample profit.`);
}

// ================================================================ 3. Task, data and models
{
  const s = content("Task, Data and Models",
    "Predict the body of the next candle at the close of the current one, then trade exactly that candle", "Models");
  header(s, L, 1.55, 6.0, "Prediction task (no look-ahead)");
  // timeline: candle t and t+1
  const ty = 2.15, cx1 = L + 0.6, cx2 = L + 3.3;
  [[cx1, "Candle t", PALE, MID], [cx2, "Candle t+1", TINT, RED]].forEach(([x, lab, fill, col]) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: ty, w: 2.2, h: 1.35, fill: { color: fill }, line: { color: col, width: 1.25 } });
    T(s, lab, { x, y: ty + 0.08, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: col, align: "center" });
  });
  T(s, "features from candles ≤ t only", { x: cx1 + 0.1, y: ty + 0.5, w: 2.0, h: 0.7, fontSize: 11, color: MID, align: "center", valign: "middle" });
  T(s, "BUY/SELL at open\nexit at close", { x: cx2 + 0.1, y: ty + 0.5, w: 2.0, h: 0.7, fontSize: 11, bold: true, color: RED_DARK, align: "center", valign: "middle" });
  s.addShape(pres.shapes.LINE, { x: cx1 + 2.2, y: ty - 0.15, w: 0, h: 1.75, line: { color: INK, width: 1.5, dashType: "dash" } });
  T(s, "decision at close of t", { x: cx1 + 1.4, y: ty + 1.62, w: 1.6, h: 0.3, fontSize: 10.5, bold: true, align: "center" });
  bullets(s, [
    [{ text: "Target: ", o: { bold: true } }, { text: "bullish if close(t+1) > open(t+1), bearish if lower; doji = miss" }],
    [{ text: "Data: ", o: { bold: true } }, { text: "NQ futures (CME E-mini Nasdaq-100, $20 per point), TradeStation 1-min bars 2006–2025" }],
    [{ text: "Splits: ", o: { bold: true } }, { text: "train 2006–Aug 2017 · validation → Jul 2021 · test Jul 2021–May 2025, purged at boundaries" }],
  ], { x: L + 0.1, y: 4.25, w: 5.9, h: 2.5, fontSize: 12 });
  header(s, 6.95, 1.55, R - 6.95, "Model families: original and optimised versions");
  const m = [
    ["Logistic regression", "weighted sum of features → probability of an up candle; simple, hard to overfit"],
    ["Random forest", "averages hundreds of decision trees built on random data subsets"],
    ["XGBoost", "trees added one by one, each correcting the previous errors; early-stopped on validation"],
    ["Neural network (MLP)", "stacked layers of neurons learning non-linear patterns; 2 to 4 hidden layers tested"],
  ];
  m.forEach(([t, b], i) => card(s, 6.95, 2.05 + i * 1.0, R - 6.95, 0.88, i + 1, t, b));
  T(s, "Features: 8 original price features (lagged returns, momentum, volatility) + 11 new ones (candle body and wicks, range, volume, realised volatility, RSI, moving-average gaps, time of day).",
    { x: 6.95, y: 6.1, w: R - 6.95, h: 0.75, fontSize: 10.5, color: MID });
  s.addNotes("Decision time is the close of candle t; the trade is the open-to-close move of candle t+1, which is exactly what the model predicts. Filling at the open right after the close is an idealised assumption, especially for one-minute candles.");
}

// ================================================================ 4. Accuracy: original vs optimised + NN depth
{
  const s = content("Original vs Optimised Models",
    "Optimisation changes out-of-sample accuracy by less than one point; deeper networks give no consistent gain", "Models");
  header(s, L, 1.55, 7.35, "Test accuracy, all candles (original → optimised, change in pp)");
  const rows = [[hdrCell("Model"), ...FREQS.map((f) => hdrCell(FREQ[f]))]];
  for (const fam of ["logreg", "random_forest", "xgboost", "mlp"]) {
    const r = [cell(FAM[fam], { bold: true })];
    for (const f of FREQS) {
      const x = D.models_all.find((q) => q.frequency === f && q.family === fam);
      const best = x.optimised === bestPred.optimised && f === bestPred.frequency;
      r.push(num(`${pct(x.orig_acc_all)} → ${pct(x.opt_acc_all)}\n${pp(x.improvement_pp)}`,
        best ? { fill: { color: TINT }, bold: true, color: RED_DARK } : {}));
    }
    rows.push(r);
  }
  const cw4 = (7.35 - 1.6) / FREQS.length;
  s.addTable(rows, tableOpts(L, 2.05, 7.35, [1.6, ...FREQS.map(() => cw4)], { rowH: [0.34, 0.62, 0.62, 0.62, 0.62], fontSize: 9.5 }));
  T(s, `Highlighted: best predictive model, ${mname(bestPred.optimised)} on ${FREQ[bestPred.frequency]} candles: ${pct(bestPredAcc.acc_nonflat)} correct on non-doji candles, ${pp(bestPred.opt_vs_majority_pp)} vs always-up (95% CI ${bestPred.opt_vs_majority_ci[0].toFixed(2)} to ${bestPred.opt_vs_majority_ci[1].toFixed(2)}). Dojis count as misses, so all-candle accuracy at 1 min is below 50%.`,
    { x: L, y: 4.95, w: 7.35, h: 0.8, fontSize: 10.5, color: MID });
  callout(s, L, 5.95, 7.35, 0.8, "Accuracy beats “always up” only at 1 and 5 minutes; from 15 minutes on, no model is reliably better");
  // NN depth chart
  const x2 = 8.3;
  header(s, x2, 1.55, R - x2, "Neural networks: test accuracy vs original 64-32-16, pp");
  const archs = ["mlp_64-32_ext", "mlp_128-64-32_ext", "mlp_256-128-64-32_ext", "mlp_128-64-32_drop0.2_bn_ext", "mlp_256-128-64-32_drop0.2_bn_ext"];
  const lbl = ["64-32", "128-64-32", "256-128-64-32", "128-64-32 +drop/BN", "256-…-32 +drop/BN"];
  const series = FREQS.map((f) => {
    const base = D.nn.find((r) => r.frequency === f && r.model === "mlp_original_64-32-16").acc_all;
    return { name: FREQ[f], labels: lbl, values: archs.map((a) => +(100 * (D.nn.find((r) => r.frequency === f && r.model === a).acc_all - base)).toFixed(2)) };
  });
  s.addChart(pres.charts.BAR, series, Object.assign({ x: x2, y: 2.0, w: R - x2, h: 3.75, barDir: "bar", barGrouping: "clustered",
    chartColors: ["E8C5CB", "D9A3AC", "C8707D", RED, RED_DARK].slice(-FREQS.length), barGapWidthPct: 40, valAxisLabelFormatCode: "+0.0;-0.0;0",
    valGridLine: { color: "E6E6E6", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "t",
    catAxisOrientation: "maxMin", catAxisLabelPos: "low" }, chartText));
  T(s, "Early stopping halted every network after 5–17 epochs: validation loss stops improving almost immediately, a sign of a very weak signal. More neurons or layers gave no consistent gain; dropout/batch norm did not help.",
    { x: x2, y: 5.85, w: R - x2, h: 0.95, fontSize: 10.5, color: MID });
  s.addNotes("Each family has an original configuration from the repository and optimised variants; the optimised one is chosen by validation balanced accuracy only, then scored once on the test period. The full metric set (precision, recall, F1, bullish and bearish accuracy, confusion matrix) is in the appendix.");
}

// ================================================================ 5. From signals to trades
{
  const s = content("From Signals to Trades",
    "Every unfiltered strategy loses money once each round trip pays $5; costs are larger than the gross edge", "Trading");
  header(s, L, 1.55, 4.6, "Trading rules (one NQ contract)");
  const steps = [
    ["Signal", "model probability that candle t+1 is bullish"],
    ["Direction", "Long & Short · Long only · Short only"],
    ["Filter (optional)", "trade only if |P(up) − 0.5| ≥ threshold chosen on validation"],
    ["Execution", "enter at open(t+1), exit at close(t+1); P&L = ±(close − open) × $20"],
    ["Costs", "$5 per round trip ($2.50 open + $2.50 close), zero slippage, 100% fills (idealised)"],
  ];
  steps.forEach(([t, b], i) => card(s, L, 2.05 + i * 0.94, 4.6, 0.82, i + 1, t, b, i === 4 ? GOLD : RED, 0.5));
  const x2 = 5.5;
  header(s, x2, 1.55, R - x2, "Validation-chosen model per candle length, Long & Short, test period");
  const rows = [[hdrCell("Candle"), hdrCell("Model"), hdrCell("Trades"), hdrCell("Gross"), hdrCell("Costs"), hdrCell("Net"),
                 hdrCell("Filtered trades"), hdrCell("Filtered net")]];
  for (const c of CAND) {
    const g = grid(c.frequency, c.model, "long_short", "none");
    rows.push([cell(FREQ[c.frequency], { bold: true }), cell(mname(c.model)), num(g.n_trades.toLocaleString("en-US")),
      num(usd(g.gross_usd), { color: g.gross_usd >= 0 ? GREEN : RED_DARK }), num(usd(-g.costs_usd)),
      num(usd(g.net_usd), { bold: true, color: RED_DARK }), num(c.test_n_trades.toLocaleString("en-US")),
      num(usd(c.test_net_usd), { bold: true, color: c.test_net_usd >= 0 ? GREEN : RED_DARK })]);
  }
  s.addTable(rows, tableOpts(x2, 2.05, R - x2, [0.68, 1.78, 0.85, 0.8, 0.78, 0.8, 0.8, 0.79], { rowH: [0.5, ...CAND.map(() => 0.44)], fontSize: 10.5 }));
  const cut = CAND.map((c) => 100 * (1 - c.test_n_trades / grid(c.frequency, c.model, "long_short").n_trades));
  T(s, `Filter thresholds (validation): ${CAND.map((c) => `${FREQ[c.frequency]} ${c.threshold}`).join(" · ")}. The filter cuts the number of trades by ${Math.min(...cut).toFixed(0)}% to ${Math.max(...cut).toFixed(1)}% and costs with it; what remains is close to zero and unreliable.`,
    { x: x2, y: 2.05 + 0.5 + 0.44 * CAND.length + 0.08, w: R - x2, h: 0.65, fontSize: 10.5, color: MID });
  const e5 = grid("5min", "reversal", "long_short");
  callout(s, x2, 5.6, R - x2, 1.15, `Example, 5-minute reversal rule: ${usd(e5.gross_usd)} gross over ${e5.n_trades.toLocaleString("en-US")} trades becomes ${usd(e5.net_usd)} after commissions: $${(e5.gross_usd / e5.n_trades).toFixed(2)} gross edge per trade vs $5.00 cost`);
  s.addNotes("High-frequency signals trade hundreds of thousands of times, so even a positive gross edge is overwhelmed by the fixed five-dollar cost per round trip. The confidence filter is the only lever that reduces costs.");
}

// ================================================================ 6. Model-by-model at the chosen candle length
{
  const s = content(`Backtest by Model: ${FREQ[F]} Candles`,
    (() => { const pos = D.models_primary.map((r) => r.optimised).concat(["reversal"]).flatMap((m) => ["long_short", "long_only", "short_only"].map((st) => grid(F, m, st))).filter((g) => g && g.net_usd > 0); return pos.length ? `After costs ${pos.length === 1 ? "only one" : pos.length} of ${3 * (D.models_primary.length + 1)} model × direction combinations ${pos.length === 1 ? "is" : "are"} positive, and only marginally` : "Every model and direction is negative after costs in the test period"; })(), "Trading");
  header(s, L, 1.55, CW, `Test period, no filter, one contract, $5 per round trip (${FREQ[F]} candles chosen on validation)`);
  const fams = D.models_primary.map((r) => r.optimised).concat(["reversal", "always_long"]);
  const rows = [[hdrCell("Model"), hdrCell("Accuracy"), hdrCell("Net: Long & Short"), hdrCell("Net: Long only"), hdrCell("Net: Short only"),
                 hdrCell("Win rate"), hdrCell("Trades"), hdrCell("Profit factor"), hdrCell("Max drawdown"), hdrCell("Net / trade")]];
  for (const m of fams) {
    const ls = grid(F, m, m === "always_long" ? "long_only" : "long_short");
    const lo = grid(F, m, "long_only"), so = grid(F, m, "short_only");
    const a = acc(F, m);
    const hl = m === C.model ? { fill: { color: TINT } } : {};
    const n = (v) => (v === undefined ? num("—", hl) : num(usd(v.net_usd), Object.assign({ color: v.net_usd >= 0 ? GREEN : RED_DARK }, hl)));
    rows.push([cell(mname(m), Object.assign({ bold: true }, hl)), num(pct(a.acc_all), hl), m === "always_long" ? num("—", hl) : n(ls), n(lo),
      m === "always_long" ? num("—", hl) : n(so), num(pct(ls.win_rate, 1), hl), num(ls.n_trades.toLocaleString("en-US"), hl),
      num(ls.profit_factor.toFixed(2), hl), num(usd(ls.max_drawdown_usd), hl), num(usd(ls.avg_net_per_trade_usd), hl)]);
  }
  s.addTable(rows, tableOpts(L, 2.05, CW, [2.55, 1.0, 1.2, 1.15, 1.15, 0.9, 0.95, 1.0, 1.15, 1.18], { rowH: 0.42, fontSize: 11 }));
  T(s, "Win rate, trades, profit factor, drawdown and net per trade refer to the Long & Short version (Long only for “always long”). Highlighted row: the model of the pre-specified best strategy.",
    { x: L, y: 5.25, w: CW, h: 0.45, fontSize: 10.5, color: MID });
  callout(s, L, 5.9, CW, 0.8, "Higher accuracy does not mean profit: the most accurate models are still down after costs, because wins and losses are of similar size and every trade pays $5");
  s.addNotes("The 1-hour candle length is where the pre-specified selection rule landed: it produced the highest validation profit. In the test period the same models lose after costs in every direction.");
}

// ================================================================ 7. Equity curves and long vs short
{
  const s = content("Equity Curves: Long vs Short",
    "Long trades dominate the losses; short-only loses least but is still negative after costs", "Trading");
  header(s, L, 1.55, 8.0, `Cumulative net P&L, ${mname(C.model)} on ${FREQ[F]} candles, test period (thousand USD)`);
  const ew = D.equity_weekly, labels = ew.dates;
  const ser = [["Long & Short", "long_short"], ["Long only", "long_only"], ["Short only", "short_only"],
               [`Filtered L&S (|p−0.5| ≥ ${C.threshold})`, "chosen"], ["Always long", "always_long"]]
    .map(([n, k]) => ({ name: n, labels, values: ew[k].map((v) => +(v / 1000).toFixed(1)) }));
  s.addChart(pres.charts.LINE, ser, Object.assign({ x: L, y: 2.0, w: 8.0, h: 4.8, chartColors: [RED, GOLD, MID, GREEN, "A6A6A6"],
    lineSize: 1.75, lineDataSymbol: "none", valAxisLabelFormatCode: "0", valGridLine: { color: "E6E6E6", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: true, legendPos: "t", catAxisLabelFrequency: 26, catAxisLabelFontSize: 9 }, chartText));
  const x2 = 8.95;
  header(s, x2, 1.55, R - x2, "Long vs short contribution (no filter)");
  const ls = grid(F, C.model, "long_short"), lo = grid(F, C.model, "long_only"), so = grid(F, C.model, "short_only");
  s.addTable([
    [hdrCell(""), hdrCell("Long only"), hdrCell("Short only")],
    [cell("Trades", { bold: true }), num(lo.n_trades.toLocaleString("en-US")), num(so.n_trades.toLocaleString("en-US"))],
    [cell("Net P&L", { bold: true }), num(usd(lo.net_usd), { color: RED_DARK }), num(usd(so.net_usd), { color: so.net_usd >= 0 ? GREEN : RED_DARK })],
    [cell("Win rate", { bold: true }), num(pct(lo.win_rate, 1)), num(pct(so.win_rate, 1))],
    [cell("Profit factor", { bold: true }), num(lo.profit_factor.toFixed(2)), num(so.profit_factor.toFixed(2))],
    [cell("Max drawdown", { bold: true }), num(usd(lo.max_drawdown_usd)), num(usd(so.max_drawdown_usd))],
  ], tableOpts(x2, 2.05, R - x2, [1.35, 1.2, 1.28], { rowH: 0.36, fontSize: 11 }));
  bullets(s, [
    `The model calls “up” ${Math.round(100 * ls.n_long / ls.n_trades)}% of the time, so Long & Short is mostly long and inherits its losses`,
    "Short-only trades far less, so it pays far less in commissions and has the smallest drawdown",
    `Gross P&L is positive (${usd(ls.gross_usd)}) but costs (${usd(-ls.costs_usd)}) exceed it; the filter cuts costs, yet the curve ends near zero`,
  ], { x: x2 + 0.05, y: 4.4, w: R - x2 - 0.05, h: 2.45, fontSize: 11.5 });
  s.addNotes("All curves are net of five dollars per round trip and plotted against calendar time (weekly). Restricting to one direction mainly changes how many trades pay costs; it does not reveal a profitable side.");
}

// ================================================================ 8. Robustness
{
  const s = content("Robustness: Costs, Walk-Forward, In vs Out of Sample",
    "Validation profits shrink or reverse on unseen data, and walk-forward retraining confirms the instability", "Trading");
  const hw = 6.1;
  header(s, L, 1.55, hw, "Test net P&L vs cost per round trip, filtered candidates (thousand USD)");
  const costs = [0, 2.5, 5, 12.5, 23];
  const ser = CAND.map((c) => ({ name: `${FREQ[c.frequency]} ${mname(c.model)}`, labels: costs.map((x) => "$" + x),
    values: costs.map((x) => +(c["test_net_at_" + (x === 2.5 ? "2.5" : x === 12.5 ? "12.5" : String(x))] / 1000).toFixed(1)) }));
  s.addChart(pres.charts.LINE, ser, Object.assign({ x: L, y: 2.0, w: hw, h: 3.0, chartColors: [RED, GOLD, MID, "C8707D", GREEN], lineSize: 2,
    lineDataSymbol: "circle", lineDataSymbolSize: 6, valAxisLabelFormatCode: "0", valGridLine: { color: "E6E6E6", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 9 }, chartText));
  T(s, "$12.50 and $23 add one tick of spread plus ¼ or 1 tick of slippage per side: realistic market-order execution.",
    { x: L, y: 5.05, w: hw, h: 0.45, fontSize: 10, color: MID });
  const x2 = L + hw + 0.4;
  header(s, x2, 1.55, R - x2, "Net P&L by sample and walk-forward years (filtered, $5)");
  const rows = [[hdrCell("Candle"), hdrCell("Train"), hdrCell("Validation"), hdrCell("Test"), hdrCell("Walk-forward\nyears > 0"), hdrCell("Walk-forward\nnet")]];
  for (const c of CAND) {
    const hl = c.overall_choice ? { fill: { color: TINT } } : {};
    rows.push([cell(FREQ[c.frequency], Object.assign({ bold: true }, hl)), num(usd(c.train_net_usd), hl), num(usd(c.val_net_usd), hl),
      num(usd(c.test_net_usd), Object.assign({ bold: true, color: c.test_net_usd >= 0 ? GREEN : RED_DARK }, hl)),
      num(`${c.wf_years_positive} of ${c.wf_years}`, hl), num(usd(c.wf_net_usd), Object.assign({ color: c.wf_net_usd >= 0 ? GREEN : RED_DARK }, hl))]);
  }
  s.addTable(rows, tableOpts(x2, 2.05, R - x2, [0.78, 0.92, 1.03, 0.98, 1.03, 0.99], { rowH: [0.55, ...CAND.map(() => 0.4)], fontSize: 10.5 }));
  T(s, `Validation picks the best of ${D.n_val_configurations} configurations, so its profit is optimistic by construction (multiple comparisons). Walk-forward: model refitted each year on all prior data, threshold re-chosen on the preceding year.`,
    { x: x2, y: 2.05 + 0.55 + 0.4 * CAND.length + 0.08, w: R - x2, h: 0.95, fontSize: 10.5, color: MID });
  callout(s, L, 5.75, CW, 1.0, `No candidate has a 95% interval of daily net P&L above zero in the test period, and none is profitable in at least 3 of 4 walk-forward years`);
  s.addNotes("Robustness checks: cost sensitivity from zero to twenty-three dollars per round trip, in-sample versus validation versus test, yearly walk-forward refits, and the threshold grid in the appendix.");
}

// ================================================================ 9. Best overall strategy
{
  const s = content("Best Overall Strategy",
    deployable.length ? "A deployable strategy was identified" : "No deployable NASDAQ trading strategy has been identified", "Conclusion");
  const ch = chSplit("test");
  const hw = (CW - 0.4) / 2;
  const box = (x, title, sub, rowsKV, color) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w: hw, h: 4.0, fill: { color: "FFFFFF" }, line: { color, width: 1.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w: hw, h: 0.75, fill: { color }, line: { color, width: 1.5 } });
    T(s, [{ text: title, options: { bold: true, breakLine: true, fontSize: 14 } }, { text: sub, options: { fontSize: 11 } }],
      { x: x + 0.2, y: 1.6, w: hw - 0.4, h: 0.75, color: "FFFFFF", valign: "middle" });
    s.addTable(rowsKV.map(([k, v, o]) => [cell(k, { bold: true, color: MID }), cell(v, Object.assign({ align: "right" }, o || {}))]),
      tableOpts(x + 0.2, 2.45, hw - 0.4, [2.3, hw - 2.7], { rowH: 0.31, fontSize: 11, border: { type: "none" } }));
  };
  box(L, "Selected by the pre-specified rule", `highest validation profit across all ${D.n_val_configurations} configurations`, [
    ["Model", mname(C.model)],
    ["Candles · direction", `${FREQ[C.frequency]} · ${STRAT[C.strategy]}`],
    ["Confidence filter", `|P(up) − 0.5| ≥ ${C.threshold}`],
    ["Test accuracy", pct(chAcc.acc_all)],
    ["Net P&L (test)", usd(ch.net_usd), { bold: true, color: ch.net_usd >= 0 ? GREEN : RED_DARK }],
    ["Trades · win rate", `${ch.n_trades.toLocaleString("en-US")} · ${pct(ch.win_rate, 1)}`],
    ["Profit factor", ch.profit_factor.toFixed(2)],
    ["Max drawdown", usd(ch.max_drawdown_usd)],
    ["Daily net 95% CI", `${usd0(ch.daily_net_ci_lo)} to ${usd0(ch.daily_net_ci_hi)}`],
    ["Walk-forward", `${C.wf_years_positive} of ${C.wf_years} years positive, ${usd(C.wf_net_usd)}`],
  ], RED);
  const bt = bestTest, btAcc = acc(bt.frequency, bt.model);
  box(L + hw + 0.4, "Best test-period result (descriptive only)", "chosen after seeing the test: not a valid selection", [
    ["Model", mname(bt.model)],
    ["Candles · direction", `${FREQ[bt.frequency]} · ${STRAT[bt.strategy]}`],
    ["Confidence filter", `|P(up) − 0.5| ≥ ${bt.threshold}`],
    ["Test accuracy", pct(btAcc.acc_all)],
    ["Net P&L (test)", usd(bt.test_net_usd), { bold: true, color: bt.test_net_usd >= 0 ? GREEN : RED_DARK }],
    ["Trades · win rate", `${bt.test_n_trades.toLocaleString("en-US")} · ${pct(bt.test_win_rate, 1)}`],
    ["Profit factor", bt.test_profit_factor.toFixed(2)],
    ["Max drawdown", usd(bt.test_max_drawdown_usd)],
    ["Daily net 95% CI", `${usd0(bt.test_daily_net_ci_lo)} to ${usd0(bt.test_daily_net_ci_hi)}`],
    ["Walk-forward", `${bt.wf_years_positive} of ${bt.wf_years} years positive, ${usd(bt.wf_net_usd)}`],
  ], GOLD);
  callout(s, L, 5.8, CW, 1.0, deployable.length ? "See criteria above." :
    `Verdict: neither strategy passes the pre-set bar (test profit at $5 and $12.50, daily-P&L interval above zero, 3 of 4 profitable walk-forward years). The ${FREQ[bt.frequency]} result also assumes filling at the next open with zero delay. Trading either would be a bet, not an edge.`);
  s.addNotes("The final slide applies the rule fixed before the results were read. Left: the strategy the rule selects. Right: the best test outcome, shown for transparency; it was picked by looking at the test period and depends on idealised one-minute fills, so it is not evidence of a deployable edge.");
}

// ================================================================ Appendix
{
  pres.addSection({ title: "Appendix" }); sec = "Appendix";
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Appendix" });
  T(s, "Appendix", { x: L + 0.05, y: 2.2, w: 8, h: 1.0, fontSize: 40, bold: true, color: "FFFFFF", valign: "bottom" });
  s.addShape(pres.shapes.LINE, { x: L + 0.05, y: 3.45, w: 1.4, h: 0, line: { color: GOLD, width: 2.5 } });
  const items = ["A1  Data, splits and sample sizes", "A2  Features, hyper-parameters and selection rules",
    "A3  Classification metrics and confusion matrix", "A4  Net P&L by candle length, model and direction",
    "A5  Why 50% means random", "A6  Reproducibility and limitations"];
  T(s, items.map((t, i) => ({ text: t, options: { breakLine: i < items.length - 1 } })),
    { x: L + 0.05, y: 3.75, w: 8, h: 3.0, fontSize: 15, color: "FFFFFF", paraSpaceAfter: 6 });
}
{
  const s = content("A1  Data, Splits and Sample Sizes", "Same chronological boundaries for every candle length; test scored once", "Appendix");
  const rows = [[hdrCell("Candle"), hdrCell("Rows (all)"), hdrCell("Train"), hdrCell("Rows fitted"), hdrCell("Validation"), hdrCell("Test"),
                 hdrCell("Test dojis"), hdrCell("Test bullish share")]];
  for (const f of FREQS) {
    const i = D.info[f], sp = i.splits, a = acc(f, "always_long");
    rows.push([cell(FREQ[f], { bold: true }), num(i.n_rows_final.toLocaleString("en-US")), num(sp.train[2].toLocaleString("en-US")),
      num(sp.train_fit_rows.toLocaleString("en-US")), num(sp.validation[2].toLocaleString("en-US")), num(sp.test[2].toLocaleString("en-US")),
      num(a.n_doji.toLocaleString("en-US")), num(pct(a.n_bullish / (a.n_bullish + a.n_bearish), 1))]);
  }
  s.addTable(rows, tableOpts(L, 1.6, CW, [1.2, 1.55, 1.5, 1.45, 1.55, 1.5, 1.4, 2.08], { rowH: 0.4, fontSize: 11 }));
  const sp = D.info[FREQS[0]].splits;
  bullets(s, [
    `Train ${sp.train[0]} → ${sp.train[1]}, validation ${sp.validation[0]} → ${sp.validation[1]}, test ${sp.test[0]} → ${sp.test[1]} (session dates). Labels crossing a boundary are purged.`,
    "Source: TradeStation continuous NQ, 1-minute bars, Europe/Rome clock, end-stamped; additively back-adjusted, so within-session point moves equal the traded contract’s.",
    "Instrument: CME E-mini Nasdaq-100 futures (NQ), $20 per index point, tick 0.25 = $5. Not MNQ ($2 per point), not QQQ, not the cash index.",
    "Candles are built from 1-minute bars inside CME sessions; a trade never spans a session close. Bad prints (|1-min return| > 2%) are quarantined causally.",
    `Training rows are non-doji candles; at 1 minute the most recent ${sp.train_fit_rows.toLocaleString("en-US")} are used. The test period was inspected with the original models in earlier project phases: results are exploratory.`,
  ], { x: L + 0.1, y: 1.6 + 0.4 * rows.length + 0.2, w: CW - 0.2, h: 6.85 - (1.6 + 0.4 * rows.length + 0.2), fontSize: 12 });
}
{
  const s = content("A2  Features, Hyper-parameters and Selection Rules", "All choices made on training and validation data; the test set never informed them", "Appendix");
  header(s, L, 1.55, 5.8, "Features (all known at the close of candle t)");
  s.addTable([
    [hdrCell("Group"), hdrCell("Features")],
    [cell("Original (8)", { bold: true }), cell("log returns lag 1, 2, 3, 5; momentum 5, 10; volatility 5, 10")],
    [cell("Candle shape", { bold: true }), cell("body / close, high-low range, upper and lower wick")],
    [cell("Volume, volatility", { bold: true }), cell("log volume vs 20-bar mean; 20-bar realised volatility")],
    [cell("Indicators", { bold: true }), cell("RSI(14) centred; gap to 20 and 50-bar moving averages")],
    [cell("Time", { bold: true }), cell("time of day within the session (sine, cosine)")],
  ], tableOpts(L, 2.05, 5.8, [1.6, 4.2], { rowH: 0.42, fontSize: 11 }));
  header(s, 6.75, 1.55, R - 6.75, "Configurations tested per candle length");
  s.addTable([
    [hdrCell("Family"), hdrCell("Original"), hdrCell("Variants (new features)")],
    [cell("Logistic", { bold: true }), cell("C = 1, 8 features"), cell("C = 1 and C = 0.01")],
    [cell("Random forest", { bold: true }), cell("200 trees, depth 6"), cell("200 trees, depth 10, leaf 200")],
    [cell("XGBoost", { bold: true }), cell("200 trees, depth 3, η 0.05"), cell("depth 3 or 5, η 0.05, early stop on validation")],
    [cell("MLP", { bold: true }), cell("64-32-16, ReLU, L2 1e-3"), cell("64-32, 128-64-32, 256-128-64-32; ±dropout 0.2 + batch norm; lr 3e-4 / batch 1024; AdamW, early stop")],
  ], tableOpts(6.75, 2.05, R - 6.75, [1.3, 1.85, R - 6.75 - 3.15], { rowH: [0.34, 0.42, 0.42, 0.5, 0.75], fontSize: 10.5 }));
  bullets(s, [
    "Optimised model per family = highest validation balanced accuracy (it may be the original).",
    "Confidence threshold per model and direction = highest validation net P&L on {0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10}, at least 200 trades.",
    "Overall strategy = highest validation net P&L; deployable only if test net > 0 at $5 and $12.50, daily-P&L 95% interval > 0 and ≥ 3 of 4 profitable walk-forward years.",
    "Scaling fitted on training rows only; network training shuffles only inside the training set; seeds fixed (42).",
  ], { x: L + 0.1, y: 4.8, w: CW - 0.2, h: 2.0, fontSize: 11.5 });
}
{
  const f = bestPred.frequency;
  const s = content("A3  Classification Metrics and Confusion Matrix", `Test period, ${FREQ[f]} candles: originals and optimised versions`, "Appendix");
  const ms = D.models_all.filter((r) => r.frequency === f).flatMap((r) => [r.original, r.optimised]).filter((v, i, a) => a.indexOf(v) === i);
  const rows = [[hdrCell("Model"), hdrCell("Accuracy"), hdrCell("Non-doji"), hdrCell("Balanced"), hdrCell("Precision (up)"), hdrCell("Recall (up)"),
                 hdrCell("F1 (up)"), hdrCell("Bullish acc."), hdrCell("Bearish acc."), hdrCell("vs majority, pp [95% CI]")]];
  for (const m of ["always_long", "reversal", ...ms]) {
    const a = acc(f, m);
    rows.push([cell(mname(m), { bold: m === bestPred.optimised }), num(pct(a.acc_all)), num(pct(a.acc_nonflat)), num(pct(a.balanced_acc_nonflat)),
      num(pct(a.precision_up)), num(pct(a.recall_up)), num(pct(a.f1_up)), num(pct(a.acc_on_bullish)), num(pct(a.acc_on_bearish)),
      num(`${a.vs_majority_pp.toFixed(2)} [${a.vs_majority_ci_lo.toFixed(2)}, ${a.vs_majority_ci_hi.toFixed(2)}]`)]);
  }
  s.addTable(rows, tableOpts(L, 1.6, CW, [2.55, 0.95, 0.95, 0.95, 1.05, 0.95, 0.9, 1.0, 1.0, 1.93], { rowH: 0.3, fontSize: 9.5 }));
  const a = acc(f, bestPred.optimised);
  const y0 = 1.6 + 0.3 * rows.length + 0.2;
  header(s, L, y0, 5.6, `Confusion matrix, ${mname(bestPred.optimised)} (non-doji candles)`);
  s.addTable([
    [hdrCell(""), hdrCell("Predicted up"), hdrCell("Predicted down")],
    [cell("Actual up", { bold: true }), num(a.tp.toLocaleString("en-US")), num(a.fn.toLocaleString("en-US"))],
    [cell("Actual down", { bold: true }), num(a.fp.toLocaleString("en-US")), num(a.tn.toLocaleString("en-US"))],
  ], tableOpts(L, y0 + 0.45, 5.6, [1.6, 2.0, 2.0], { rowH: 0.3, fontSize: 10.5 }));
  T(s, `n = ${a.n_all.toLocaleString("en-US")} test candles (${a.n_bullish.toLocaleString("en-US")} bullish, ${a.n_bearish.toLocaleString("en-US")} bearish, ${a.n_doji.toLocaleString("en-US")} doji). Accuracy ± ${(100 * a.acc_all_ci95).toFixed(2)} pp (binomial 95%); the paired comparison with the majority rule uses 20-session block bootstrap.`,
    { x: 6.6, y: y0 + 0.5, w: R - 6.6, h: 1.1, fontSize: 11, color: MID });
}
{
  const s = content("A4  Net P&L by Candle Length, Model and Direction", "Test period, $5 per round trip, thousand USD: unfiltered (left) and with the validation-chosen filter (right)", "Appendix");
  const fam = ["logreg", "random_forest", "xgboost", "mlp"];
  const rows = [[hdrCell("Candle"), hdrCell("Model (optimised)"), hdrCell("L&S"), hdrCell("Long"), hdrCell("Short"), hdrCell("L&S filtered"), hdrCell("Long filtered"), hdrCell("Short filtered")]];
  const k = (v) => (v === undefined ? num("—") : num((v.net_usd / 1000).toFixed(1), { color: v.net_usd >= 0 ? GREEN : RED_DARK }));
  const gv = (f, m, st, tk) => D.strategy_grid_test.find((r) => r.frequency === f && r.model === m && r.strategy === st && r.threshold_kind === tk);
  for (const f of FREQS) for (const fm of fam) {
    const m = D.models_all.find((r) => r.frequency === f && r.family === fm).optimised;
    rows.push([cell(fm === "logreg" ? FREQ[f] : "", { bold: true }), cell(mname(m)),
      ...["long_short", "long_only", "short_only"].map((st) => k(gv(f, m, st, "none"))),
      ...["long_short", "long_only", "short_only"].map((st) => k(gv(f, m, st, "validated")))]);
  }
  s.addTable(rows, tableOpts(L, 1.6, CW, [1.0, 3.0, 1.3, 1.3, 1.3, 1.45, 1.45, 1.43], { rowH: FREQS.length > 4 ? 0.228 : 0.265, fontSize: FREQS.length > 4 ? 8.5 : 9.5 }));
  T(s, "“—”: the validation-chosen threshold was 0 (no filter) or produced fewer than 200 validation trades. Filtered results are spread around zero with no consistent sign across candle lengths or models.",
    { x: L, y: 6.5, w: CW, h: 0.4, fontSize: 10, color: MID });
}
{
  const s = content("A5  Why 50% Means Random", "Predicting the next candle is close to tossing a coin; only results outside luck’s range count", "Appendix");
  header(s, L, 1.55, 6.1, "How far luck alone moves a fair coin’s accuracy (95% range, ± pp)");
  const ns = [100, 928, 10000, 100000, 1370000];
  s.addChart(pres.charts.BAR, [{ name: "± pp", labels: ["100", "1,000", "10,000", "100,000", "1.37M"],
    values: ns.map((n) => +(196 * Math.sqrt(0.25 / n)).toFixed(2)) }], Object.assign({ x: L, y: 2.0, w: 6.1, h: 3.4, barDir: "col",
    chartColors: [RED], barGapWidthPct: 50, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "±0.00",
    valAxisMinVal: 0, valAxisMaxVal: 11, valAxisMajorUnit: 2, valAxisLabelFormatCode: "0", valGridLine: { color: "E6E6E6", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false }, chartText));
  T(s, "Range = 1.96 × √(0.25 / n). With few trades a coin can look like a 55–60% “model”; with a million it cannot leave 50 ± 0.1%.",
    { x: L, y: 5.45, w: 6.1, h: 0.55, fontSize: 10.5, color: MID });
  header(s, 7.05, 1.55, R - 7.05, "Reading accuracy around 50%");
  const rows = [
    ["Coin toss = 50%", "if up and down are equally likely and unpredictable, any rule is right half the time"],
    ["Dojis lower the bar", "candles with close = open count as misses, so all-candle accuracy can sit below 50%"],
    ["Imbalance is not skill", "NQ rose over time, so “always up” already beats a coin without predicting anything"],
    ["Accuracy ≠ money", "a 51% hit rate earns nothing if wins and losses are equal in size and each trade costs $5"],
  ];
  rows.forEach(([t, b], i) => card(s, 7.05, 2.05 + i * 1.0, R - 7.05, 0.88, i + 1, t, b));
  callout(s, L, 6.15, CW, 0.6, `Best model: ${pct(bestPredAcc.acc_nonflat)} on non-doji ${FREQ[bestPred.frequency]} candles: statistically real, economically too small`);
}
{
  const s = content("A6  Reproducibility and Limitations", "Every number in this deck is regenerated and cross-checked by four commands from the public repository", "Appendix");
  header(s, L, 1.55, 6.0, "Reproduce (Windows PowerShell, project folder)");
  T(s, [
    { text: "Repository: ", options: { bold: true } },
    { text: "github.com/Paciooo/Project_ai_ML_bigdata", options: { hyperlink: { url: REPO }, color: RED, underline: { style: "sng" }, breakLine: true } },
    { text: '$env:FUTURES_DATA_DIR="C:\\path\\to\\futures-data"', options: { fontFace: "Courier New", breakLine: true } },
    { text: "1. python study.py", options: { fontFace: "Courier New", breakLine: true } },
    { text: "    trains all models, backtests, walk-forward (about 15 min on 16 CPU cores, no GPU)", options: { color: MID, breakLine: true } },
    { text: "2. python report.py results/nq_study/runs/<run> presentation/deck_data.json", options: { fontFace: "Courier New", breakLine: true } },
    { text: "3. python verify.py results/nq_study/runs/<run>", options: { fontFace: "Courier New", breakLine: true } },
    { text: "4. cd presentation; npm install; node build_deck.js", options: { fontFace: "Courier New", breakLine: true } },
    { text: `Run used here: ${D.run_id} (commit ${D.run.commit ? D.run.commit.slice(0, 7) : "local"}${D.run.dirty ? ", uncommitted changes" : ""}), seed ${D.args.seed}, latency ${D.args.latency}, $${D.args.cost} per round trip.`, options: { color: MID } },
  ], { x: L + 0.1, y: 2.05, w: 5.9, h: 3.2, fontSize: 11.5, paraSpaceAfter: 6 });
  header(s, 6.95, 1.55, R - 6.95, "Limitations");
  bullets(s, [
    "Idealised execution: fill at the next open with zero delay and zero slippage, 100% limit fills; a live limit order may not fill.",
    "Bar prices, not bid/ask quotes; costs are assumptions. $12.50 / $23 scenarios approximate market orders.",
    "Continuous back-adjusted series; roll dates are approximate (no position spans a session).",
    "The 2021–2025 test period was seen in earlier phases; data after May 2025 (reserved) was not available.",
    "Daily Sharpe ratios use 252 sessions; percentages use an assumed $100,000 account.",
  ], { x: 7.05, y: 2.05, w: R - 7.05, h: 4.7, fontSize: 11.5 });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
