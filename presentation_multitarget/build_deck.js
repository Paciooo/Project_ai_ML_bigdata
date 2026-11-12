// Multi-target NASDAQ deck (8 slides). Every number comes from deck_data.json, written by
//   python multitarget_report.py results/multitarget/runs/<run id> presentation_multitarget
// Build: node build_deck.js NASDAQ_multitarget_report.pptx   (needs pptxgenjs; NODE_PATH may point
//        to ../presentation/node_modules)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || "NASDAQ_multitarget_report.pptx";
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const IMG = (f) => path.join(__dirname, f);

const RED = "8E1728", RED_DARK = "5E0F1B", TINT = "F6E7E9", GOLD = "B8964B", GREEN = "2E6B3F", BLUE = "1F5FA8";
const INK = "1A1A1A", GREY = "7F7F7F", MID = "595959", LINE = "BFBFBF", PALE = "F4F4F4";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.author = "Alessandro Pacifici";
pres.title = "What Can Machine Learning Predict in NASDAQ Markets?";
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
    { placeholder: { options: { name: "subtitle", type: "body", x: L, y: 0.88, w: CW, h: 0.45, fontSize: 15, color: GREY,
                                margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: { x: L, y: 7.03, w: 0.6, h: 0.3, fontSize: 9, color: GREY, margin: 0 },
});
pres.defineSlideMaster({ title: "COVER", background: { color: RED } });

function content(title, subtitle) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  s.addText(subtitle, { placeholder: "subtitle" });
  return s;
}
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: "Arial", color: INK, fontSize: 12,
                                                          margin: 0, valign: "top" }, o));
function header(s, x, y, w, text, color = MID) {
  T(s, text, { x: x + 0.05, y, w: w - 0.05, h: 0.3, fontSize: 12.5, bold: true, color, valign: "bottom" });
  s.addShape(pres.shapes.LINE, { x, y: y + 0.36, w, h: 0, line: { color: GREY, width: 1 } });
}
function bullets(s, items, o) {
  const runs = items.map((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    return parts.map((p, j) => ({ text: p.text, options: Object.assign({
      bullet: j === 0 ? { code: "25AA" } : undefined, color: INK,
      breakLine: j === parts.length - 1 && i < items.length - 1, paraSpaceAfter: 6 }, p.o || {}) }));
  }).flat();
  T(s, runs, Object.assign({ fontSize: 12 }, o));
}
function callout(s, x, y, w, h, text, fontSize = 12) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: TINT }, line: { color: RED, width: 1, dashType: "dash" } });
  T(s, text, { x: x + 0.15, y, w: w - 0.3, h, fontSize, bold: true, color: RED_DARK, align: "center", valign: "middle" });
}
const hdr = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, color: "FFFFFF", fill: { color: RED }, align: "center", valign: "middle" }, o) });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ valign: "middle" }, o) });
const num = (t, o = {}) => cell(t, Object.assign({ align: "center" }, o));
const topts = (x, y, w, colW, extra = {}) => Object.assign({ x, y, w, colW, fontFace: "Arial", fontSize: 11, color: INK,
  border: { type: "solid", pt: 0.75, color: LINE }, margin: [3, 5, 3, 5], rowH: 0.3 }, extra);
const assume = (s) => T(s, "Trading: idealised fills at the next open, zero latency, 100% fills, zero slippage, $2.50 per round trip, 1 NQ = $20/point",
  { x: L + 0.7, y: 7.03, w: 8.0, h: 0.3, fontSize: 9, color: GREY });

// ---------------------------------------------------------------- data helpers
const FREQ = { "5min": "5 min", "15min": "15 min", "30min": "30 min", "1h": "1 hour", session: "1 day" };
const FREQS = ["5min", "15min", "30min", "1h", "session"].filter((f) => D.dataset[f]);
const MODEL = { zero: "Zero return", train_mean: "Training mean", majority: "Majority class", persistence: "Persistence",
  rolling_mean_22: "Rolling mean (22)", har_ols: "HAR-OLS", ridge: "Ridge", logistic: "Logistic", random_forest: "Random forest",
  xgboost: "XGBoost", "mlp_64-32-16": "DNN 64-32-16", "mlp_64-32": "DNN 64-32", "mlp_128-64-32": "DNN 128-64-32",
  "mlp_256-128-64-32": "DNN 256-128-64-32", timestral: "Timestral-5M (Mistral)" };
const TASK = { return: "Return", direction: "Direction", volatility: "Volatility" };
const SEL = D.selected;
const byTask = (t) => SEL.filter((r) => r.task === t);
const f2 = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const pctv = (v, d = 1) => (100 * v).toFixed(d) + "%";
const ciTxt = (lo, hi, d = 2, u = "") => `[${f2(lo, d)}${u}, ${f2(hi, d)}${u}]`;
const sig = (r) => r.ci_lo > 0 ? "interval above zero" : r.ci_hi < 0 ? "interval below zero" : "not distinguishable from zero";
function usd(v) {
  const a = Math.abs(v), sign = v < 0 ? "−" : "+";
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${sign}$${(a / 1e3).toFixed(1)}k`;
  return `${sign}$${a.toFixed(0)}`;
}
const cfgName = (r) => `${MODEL[r.model]}, ${FREQ[r.frequency]}`;
const RET = byTask("return"), DIR = byTask("direction"), VOL = byTask("volatility");
const bestDir = DIR[0], bestRet = RET[0], bestVol = VOL[0];
const trBest = D.trading[String(bestDir.id)];
const trade = (st) => trBest.find((t) => t.strategy === st);
const TS = D.timestral_test;
const volR2 = (r) => `${f2(r.t_r2_log_vs_persistence_pct, 0)}%`;
const tsBeats = Object.entries(TS).filter(([f, v]) => v.vol_r2_log_pct > 0).map(([f]) => FREQ[f]);

// ================================================================ 1. Title
{
  const s = pres.addSlide({ masterName: "COVER" });
  T(s, "What Can Machine Learning Predict in NASDAQ Markets?", { x: 0.8, y: 1.75, w: 11.7, h: 1.5, fontSize: 38, bold: true, color: "FFFFFF", valign: "bottom" });
  T(s, "Price, Direction and Volatility Forecasting Across Models and Timeframes", { x: 0.8, y: 3.4, w: 11.7, h: 0.6, fontSize: 20, color: "F3D9DD" });
  s.addShape(pres.shapes.LINE, { x: 0.8, y: 4.25, w: 3.2, h: 0, line: { color: GOLD, width: 2 } });
  T(s, [{ text: "Alessandro Pacifici", options: { bold: true, breakLine: true } },
        { text: "AI, Machine Learning and Big Data for Finance and Management · Waseda University", options: { breakLine: true } },
        { text: "Final report · November 2026 · E-mini Nasdaq-100 futures (NQ), 2006–2025" }],
    { x: 0.8, y: 4.45, w: 11.7, h: 1.2, fontSize: 14, color: "FFFFFF" });
}

// ================================================================ 2. Executive summary
{
  const s = content("Executive Summary", "Volatility is clearly predictable one candle ahead; returns and candle direction are barely so");
  const retR = bestRet, dirR = bestDir;
  const volDirTxt = VOL.map((r) => `${pctv(r.t_voldir_acc)} vs ${pctv(r.t_voldir_majority_acc)} (${FREQ[r.frequency]})`).join("; ");
  const blocks = [
    ["Problem", "Market moves are hard to forecast, but not every characteristic of the market need be equally hard. Which one — the next return, the next candle's direction or the next candle's realized volatility — can machine learning actually predict?"],
    ["Method", `Nasdaq-100 futures (NQ), 1-minute data 2006–2025, rebuilt into ${FREQS.length} candle lengths (5 min to 1 day). ${D.n_models_tested} models, from naive benchmarks to deep networks and Mistral's Timestral, trained on 2006–2017, tuned and selected on 2017–2021, tested once on 2021–2025.`],
    ["Innovation", "One pipeline, one data set and one test period for three different targets: the same models are compared across tasks and timeframes, so the answer to \"what is predictable\" does not depend on a different set-up per task."],
    ["Main findings", `Volatility: ML explains ${VOL.map((r) => `${volR2(r)} (${FREQ[r.frequency]})`).join(" and ")} of the error left by persistence; RV UP/DOWN accuracy ${volDirTxt}. Return: R² ${f2(retR.t_r2_oos_pct, 2)}% vs the zero forecast. Direction: ${pctv(dirR.t_accuracy, 2)} vs ${pctv(dirR.t_majority_acc, 2)} majority (${cfgName(dirR)}), but Long & Short ${usd(trade("long_short").net_usd)} after $2.50 costs. Mistral's Timestral (zero-shot) beat no benchmark${tsBeats.length ? ` except volatility at ${tsBeats.join(", ")}` : ""}.`],
    ["Usefulness", "Volatility forecasts serve risk limits, position sizing, stop distances and volatility-sensitive decisions (e.g. whether to hold long or short straddles). Direction and return forecasts at this horizon are too weak to support trading after costs."],
    ["Conclusion", "Realized volatility is by far the most predictable characteristic. The evidence for direction and returns is weak and fragile. The test period was already seen in earlier project phases, so all results are exploratory."],
  ];
  const cw = (CW - 0.4) / 2, ch = 1.65;
  blocks.forEach(([h, b], i) => {
    const x = L + (i % 2) * (cw + 0.4), y = 1.6 + Math.floor(i / 2) * (ch + 0.13);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.08, h: ch, fill: { color: i === 5 ? GOLD : RED }, line: { type: "none" } });
    T(s, [{ text: h, options: { bold: true, color: RED, fontSize: 13.5, breakLine: true } },
          { text: b, options: { fontSize: 11.5, color: INK } }], { x: x + 0.2, y: y + 0.02, w: cw - 0.25, h: ch - 0.04, valign: "top" });
  });
}

// ================================================================ 3. Data & methodology
{
  const s = content("Data & Methodology", "One instrument, five candle lengths, three targets, one chronological split");
  // workflow
  const steps = ["NASDAQ historical data\nNQ 1-min, 2006–2025", "Feature engineering\nlags, momentum, volatility, time of day", "Models\nbenchmarks · ML · DNN · Timestral", "Forecasts\nfor candle t+1", "Out-of-sample evaluation\ntest 2021-07 → 2025-05"];
  const bw = (CW - 4 * 0.3) / 5;
  steps.forEach((t, i) => {
    const x = L + i * (bw + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: bw, h: 0.85, rectRadius: 0.06, fill: { color: i === 4 ? RED : TINT }, line: { color: RED, width: 1 } });
    const [a, b] = t.split("\n");
    T(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b, options: { fontSize: 10 } }],
      { x: x + 0.08, y: 1.6, w: bw - 0.16, h: 0.85, fontSize: 11.5, align: "center", valign: "middle", color: i === 4 ? "FFFFFF" : INK });
    if (i < 4) s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + bw + 0.08, y: 1.93, w: 0.15, h: 0.2, rotate: 90, fill: { color: RED }, line: { type: "none" } });
  });
  // targets
  header(s, L, 2.65, 6.1, "Three targets, all for the next candle (t+1), decided at the close of t");
  const tg = [
    ["1  Next return", "r = log(C(t+1)/C(t)), regression", "MAE, RMSE, out-of-sample R² vs zero forecast"],
    ["2  Next-candle direction", "UP if C(t+1) > O(t+1); doji kept, counts as a miss", "accuracy, balanced accuracy, F1 vs majority class"],
    ["3  Future realized volatility", "RV = Σ (1-min log returns)² inside candle t+1", "log-RV error (R²) vs persistence; RV UP if RV(t+1) > RV(t)"],
  ];
  s.addTable([[hdr("Target"), hdr("Definition"), hdr("Evaluated with")], ...tg.map(([a, b, c]) => [cell(a, { bold: true }), cell(b), cell(c)])],
    topts(L, 3.12, 6.1, [1.75, 2.3, 2.05], { fontSize: 10, rowH: 0.42 }));
  bullets(s, [
    [{ text: "Instrument: ", o: { bold: true } }, { text: "E-mini Nasdaq-100 futures (NQ, CME), back-adjusted continuous series, $20 per point. No QQQ, MNQ, cash index or VIX." }],
    [{ text: "Leakage control: ", o: { bold: true } }, { text: "features use candles ≤ t only; scaling fitted on training data; labels never cross a session or a split boundary (purged)." }],
    [{ text: "Selection: ", o: { bold: true } }, { text: "hyper-parameters, early stopping and the five configurations chosen on validation only; test scored once (volatility score revised once, slide 7)." }],
  ], { x: L, y: 4.95, w: 6.1, h: 1.95, fontSize: 11 });
  // timeframe table
  const x2 = L + 6.45, w2 = CW - 6.45;
  header(s, x2, 2.65, w2, "Candles per timeframe (rows with a valid next candle)");
  const rows = FREQS.map((f) => { const d = D.dataset[f];
    return [num(FREQ[f], { bold: true }), num(d.train[2].toLocaleString("en-US")), num(d.validation[2].toLocaleString("en-US")), num(d.test[2].toLocaleString("en-US"))]; });
  const d0 = D.dataset[FREQS[0]];
  s.addTable([[hdr("Candle"), hdr(`Train\n${d0.train[0].slice(0, 7)} → ${d0.train[1].slice(0, 7)}`), hdr(`Validation\n${d0.validation[0].slice(0, 7)} → ${d0.validation[1].slice(0, 7)}`), hdr(`Test\n${d0.test[0].slice(0, 7)} → ${d0.test[1].slice(0, 7)}`)], ...rows],
    topts(x2, 3.12, w2, [1.15, (w2 - 1.15) / 3, (w2 - 1.15) / 3, (w2 - 1.15) / 3], { fontSize: 10.5, rowH: 0.34 }));
  T(s, `Features (${D.dataset[FREQS[0]].features.length}): past returns and momentum, rolling volatility, range, body, wicks, RSI, moving-average gaps, relative volume, time of day, and log realized variance of the last 1, 5 and 22 candles plus the same time slot on the 5 previous sessions. Timestral sees only the last 512 closes. 1-minute candles excluded: their realized variance would be a single squared return.`,
    { x: x2, y: 5.35, w: w2, h: 1.5, fontSize: 10.5, color: MID });
}

// ================================================================ 4. All models tested
{
  const s = content("All Models Tested", `${D.n_models_tested} models on ${FREQS.length} timeframes: ${D.n_configs} model–timeframe–task configurations, all run on the same data`);
  const fam = [
    ["Baselines", [["Zero return / training mean", "naive forecast", "R"], ["Majority class", "always the most frequent training class", "D"], ["Persistence", "RV(t+1) = RV(t)", "V"], ["Rolling mean (22 candles)", "average past variance", "V"]]],
    ["Statistical", [["HAR-OLS", "linear regression on past log RV (1, 5, 22 candles, same time slot)", "V"], ["Ridge regression", "L2-penalised linear, α chosen on validation", "R · V"], ["Logistic regression", "L2-penalised, C chosen on validation", "D"]]],
    ["Machine learning", [["Random forest", "200 trees, depth 8", "R · D · V"], ["XGBoost", "gradient boosting, depth 4, early stopping", "R · D · V"]]],
    ["Deep learning", [["DNN 64-32-16 (original)", "feed-forward, ReLU, AdamW, early stopping", "R · D · V"], ["DNN 64-32", "", "R · D · V"], ["DNN 128-64-32", "", "R · D · V"], ["DNN 256-128-64-32", "", "R · D · V"]]],
    ["Mistral AI", [["Timestral-5M (pretrained, zero-shot)", "forecasts quantiles of the next close from 512 past closes; median → return & direction, q10–q90 spread → volatility (calibrated on validation)", "R · D · V"]]],
  ];
  const body = [[hdr("Family"), hdr("Model"), hdr("Method"), hdr("Tasks")]];
  fam.forEach(([f, ms]) => ms.forEach((m, i) => body.push([
    i === 0 ? cell(f, { bold: true, color: RED, rowspan: ms.length, fill: { color: PALE } }) : null,
    cell(m[0], { bold: true }), cell(m[1], { color: MID }), num(m[2])].filter((c) => c !== null))));
  s.addTable(body, topts(L, 1.55, 8.6, [1.45, 2.55, 3.75, 0.85], { fontSize: 10, rowH: 0.29 }));
  const x2 = L + 8.9, w2 = CW - 8.9;
  header(s, x2, 1.5, w2, "Notes");
  bullets(s, [
    [{ text: "R", o: { bold: true } }, { text: " return · " }, { text: "D", o: { bold: true } }, { text: " direction · " }, { text: "V", o: { bold: true } }, { text: " volatility" }],
    [{ text: "Mistral's role: ", o: { bold: true } }, { text: "Timestral is a time-series foundation model (local weights, no API, no training here). It is not a language model and it does not read news. Scored on 20,000 random validation and 20,000 test origins per timeframe (all daily ones)." }],
    [{ text: "Timestral outcome (test): ", o: { bold: true } }, { text: `return R² ${Object.values(TS).map((v) => f2(v.return_r2_pct, 1)).join(" / ")}% and direction ${Object.values(TS).map((v) => f2(v.direction_diff_pp, 1)).join(" / ")} pp vs majority (5 min → 1 day); volatility better than persistence only at ${tsBeats.join(", ") || "no timeframe"}. Not among the five.` }],
    [{ text: "Considered, not run: ", o: { bold: true } }, { text: "LSTM (the feature networks already reach their early-stopping optimum within a few epochs), larger Timestral sizes (28M–209M need new downloads)." }],
    [{ text: "Not used: ", o: { bold: true } }, { text: "Mistral chat / LLM API — no such integration exists in the repository." }],
  ], { x: x2, y: 1.98, w: w2, h: 4.9, fontSize: 10.5 });
}

// ================================================================ 5. Return forecasting
{
  const s = content("Price / Return Forecasting", RET.every((r) => r.ci_lo <= 0) ? "Can ML beat the zero-return forecast? Out of sample, no: its error equals that of forecasting zero"
    : "Can ML beat the zero-return forecast? Only by a tiny margin");
  const rows = RET.map((r) => [num(String(r.id), { bold: true }), cell(cfgName(r), { bold: true }), num(r.t_mae_bps.toFixed(2)), num(r.t_rmse_bps.toFixed(2)),
    num(r.t_rmse_zero_bps.toFixed(2)), num(`${f2(r.t_r2_oos_pct, 3)}%`, { bold: true, color: r.ci_lo > 0 ? GREEN : INK }), num(ciTxt(r.ci_lo, r.ci_hi, 3, "%")), num(pctv(r.t_dir_acc))]);
  s.addTable([[hdr("#"), hdr("Model, timeframe"), hdr("MAE (bp)"), hdr("RMSE (bp)"), hdr("RMSE zero (bp)"), hdr("R² vs zero"), hdr("95% interval"), hdr("Sign right")], ...rows],
    topts(L, 1.6, 6.6, [0.3, 1.9, 0.6, 0.65, 0.8, 0.75, 1.0, 0.6], { fontSize: 10, rowH: 0.4 }));
  s.addImage({ path: IMG(D.charts[String(bestRet.id)]), x: L + 6.9, y: 1.55, w: CW - 6.9, h: (CW - 6.9) * 3.6 / 5.6 });
  const n = RET.map((r) => `${cfgName(r)}: R² ${f2(r.t_r2_oos_pct, 3)}%, interval ${sig(r)}`).join("; ");
  bullets(s, [
    [{ text: "Result: ", o: { bold: true } }, { text: `${n}. A positive R² would mean a smaller squared error than always forecasting zero.` }],
    [{ text: "Predicted vs realized (right): ", o: { bold: true } }, { text: `realized returns per forecast decile. The forecasts span less than ±0.5 bp, while a typical candle moves about ${bestRet.t_rmse_zero_bps.toFixed(1)} bp (RMSE): the signal is buried in noise.` }],
    [{ text: "Economic reading: ", o: { bold: true } }, { text: "no usable return signal: at these horizons NQ returns behave close to a random walk, as efficient-market reasoning suggests. Every return model and Timestral were tested on all five timeframes (inventory, slide 4)." }],
  ], { x: L, y: 2.75, w: 6.6, h: 3.6, fontSize: 11.5 });
  T(s, "bp = basis point = 0.01%. Intervals: 95% moving-block bootstrap over 20-session blocks. Test 2021-07 → 2025-05 (exploratory: seen in earlier project phases).",
    { x: L + 6.9, y: 5.55, w: CW - 6.9, h: 0.9, fontSize: 9.5, color: GREY });
}

// ================================================================ 6. Direction & trading
{
  const s = content("Direction Classification & Trading", trade("long_short").net_usd < 0
    ? "Slightly better than the majority class, but the edge per trade is smaller than the commission"
    : "Slightly better than the majority class; the trading result is fragile");
  const rows = DIR.map((r) => [num(String(r.id), { bold: true }), cell(cfgName(r), { bold: true }), num(pctv(r.t_accuracy, 2), { bold: true }),
    num(pctv(r.t_balanced_acc, 2)), num(pctv(r.t_majority_acc, 2)), num(`${f2(r.t_diff_pp)} pp`, { color: r.ci_lo > 0 ? GREEN : INK }), num(ciTxt(r.ci_lo, r.ci_hi)), num(r.t_f1_up.toFixed(3))]);
  s.addTable([[hdr("#"), hdr("Model, timeframe"), hdr("Accuracy"), hdr("Balanced acc."), hdr("Majority"), hdr("Gain"), hdr("95% interval"), hdr("F1 up")], ...rows],
    topts(L, 1.6, 6.3, [0.3, 1.55, 0.8, 0.8, 0.75, 0.7, 0.9, 0.5], { fontSize: 9.5, rowH: 0.4 }));
  const c = bestDir;
  T(s, `Confusion matrix, ${cfgName(c)} (non-doji test candles): predicted UP & was UP ${c.t_tp.toLocaleString("en-US")} · UP but DOWN ${c.t_fp.toLocaleString("en-US")} · DOWN & DOWN ${c.t_tn.toLocaleString("en-US")} · DOWN but UP ${c.t_fn.toLocaleString("en-US")}; dojis ${c.t_n_doji.toLocaleString("en-US")}.`,
    { x: L, y: 1.65 + 0.4 * (DIR.length + 1), w: 6.3, h: 0.55, fontSize: 9.5, color: MID });
  const ty = 2.4 + 0.4 * (DIR.length + 1);
  header(s, L, ty, 6.3, `Trading rule, ${cfgName(c)}: UP → long, DOWN → short, next open → close`);
  const S = { long_short: "Long & Short", long_only: "Long only", short_only: "Short only" };
  const tr = ["long_short", "long_only", "short_only"].map((st) => { const t = trade(st);
    return [cell(S[st], { bold: true }), num(t.n_trades.toLocaleString("en-US")), num(usd(t.net_usd), { bold: true, color: t.net_usd > 0 ? GREEN : RED }),
      num(pctv(t.win_rate)), num(t.profit_factor ? t.profit_factor.toFixed(2) : "–"), num(usd(t.max_drawdown_usd)), num((t.avg_net_per_trade_usd < 0 ? "−$" : "+$") + Math.abs(t.avg_net_per_trade_usd).toFixed(2)), num(usd(t.net_usd_at_5))]; });
  s.addTable([[hdr("Strategy"), hdr("Trades"), hdr("Net P&L"), hdr("Win rate"), hdr("Profit factor"), hdr("Max DD"), hdr("Net / trade"), hdr("Net at $5")], ...tr],
    topts(L, ty + 0.45, 6.3, [1.1, 0.7, 0.8, 0.65, 0.7, 0.8, 0.7, 0.85], { fontSize: 10, rowH: 0.34 }));
  s.addImage({ path: IMG(D.charts[String(bestDir.id)]), x: L + 6.6, y: 1.5, w: CW - 6.6, h: (CW - 6.6) * 3.6 / 6.4 });
  const ls = trade("long_short");
  bullets(s, [
    [{ text: "Reading: ", o: { bold: true } }, { text: `accuracy gain ${f2(c.t_diff_pp)} pp (${sig(c)}). Before costs Long & Short earns ${usd(ls.gross_usd)} on ${ls.n_trades.toLocaleString("en-US")} trades, $${(ls.gross_usd / ls.n_trades).toFixed(2)} per trade; the $2.50 round trip turns it into ${usd(ls.net_usd)}.` }],
    [{ text: "Accuracy ≠ profit: ", o: { bold: true } }, { text: "with moves of size W, a hit rate p earns (2p − 1)·W − cost per trade; at p ≈ 51% and a 5-minute W of a few points the edge is cents, the cost is dollars. Fills are idealised." }],
  ], { x: L + 6.6, y: 1.6 + (CW - 6.6) * 3.6 / 6.4 + 0.1, w: CW - 6.6, h: 2.0, fontSize: 11 });
  assume(s);
}

// ================================================================ 7. Volatility
{
  const s = content("Volatility Forecasting", "Next-candle realized volatility is strongly predictable, well beyond persistence");
  const rows = VOL.map((r) => [num(String(r.id), { bold: true }), cell(cfgName(r), { bold: true }), num(r.t_rmse_log.toFixed(3), { bold: true }),
    num(r.t_rmse_log_persistence.toFixed(3)), num(r.har_rmse_log != null ? r.har_rmse_log.toFixed(3) : "–"),
    num(`${f2(r.t_r2_log_vs_persistence_pct, 1)}%\n${ciTxt(r.ci_lo, r.ci_hi, 1)}`, { color: GREEN, bold: true }),
    num(r.har_qlike != null ? `${r.t_qlike.toFixed(3)} / ${r.har_qlike.toFixed(3)}` : r.t_qlike.toFixed(3)),
    num(pctv(r.t_voldir_acc), { bold: true }), num(pctv(r.t_voldir_majority_acc)), num(pctv(r.t_voldir_balanced_acc))]);
  s.addTable([[hdr("#"), hdr("Model, timeframe"), hdr("RMSE log RV"), hdr("Persist."), hdr("HAR-OLS"), hdr("R² vs persist."), hdr("QLIKE model / HAR"), hdr("RV dir. acc."), hdr("Majority"), hdr("Bal. acc.")], ...rows],
    topts(L, 1.6, 7.6, [0.3, 1.7, 0.65, 0.7, 0.75, 0.8, 0.65, 0.7, 0.7, 0.65], { fontSize: 9.5, rowH: 0.4 }));
  s.addImage({ path: IMG(D.charts[String(bestVol.id)]), x: L + 7.85, y: 1.5, w: CW - 7.85, h: (CW - 7.85) * 3.6 / 5.6 });
  const yb = 1.6 + 0.6 + 0.68 * VOL.length + 0.15;
  bullets(s, [
    [{ text: "Method: ", o: { bold: true } }, { text: "realized variance of the next candle from its own 1-minute squared log returns (r² is a variance proxy; volatility = √variance). Models forecast log RV; R² = share of persistence's squared error removed. QLIKE (variance level, lower is better) shown as a check." }],
    [{ text: "Why it works: ", o: { bold: true } }, { text: `volatility clusters and mean-reverts, and intraday it follows a strong time-of-day pattern. The linear HAR-OLS model already removes ${VOL.map((r) => `${r.har_r2_log_pct.toFixed(0)}%`).join(" / ")}; the ML models add ${VOL.map((r) => `${(r.t_r2_log_vs_persistence_pct - r.har_r2_log_pct).toFixed(0)}`).join(" / ")} points.` }],
    [{ text: "Direction of volatility: ", o: { bold: true } }, { text: `RV UP if RV(t+1) > RV(t). ${VOL.map((r) => `${cfgName(r)}: ${pctv(r.t_voldir_acc)} vs ${pctv(r.t_voldir_majority_acc)} majority`).join("; ")}.` }],
  ], { x: L, y: yb, w: 7.6, h: 6.9 - yb, fontSize: 11 });
  T(s, "Selection score changed after a first report: QLIKE vs persistence was dominated on validation by a few limit-locked candles (March 2020) with ~zero variance, so log-RV squared error is used. Interval: 95% block bootstrap.",
    { x: L, y: 6.35, w: 7.6, h: 0.55, fontSize: 9, color: GREY });
  callout(s, L + 7.85, 4.75, CW - 7.85, 2.05,
    "Options relevance (qualitative): rising realized volatility favours long straddles / strangles, falling volatility favours short ones. But option prices embed IMPLIED volatility: a correct realized-volatility forecast is not a profitable options trade unless it beats what the market already prices. No option prices or P&L were computed.", 10.5);
}

// ================================================================ 8. Final comparison
{
  const s = content("Which NASDAQ Market Characteristic Is Most Predictable?", "Five configurations selected on validation, scored once on the 2021–2025 test");
  const rows = SEL.map((r) => [num(String(r.id), { bold: true }), cell(TASK[r.task], { bold: true, color: RED }), cell(MODEL[r.model]), num(FREQ[r.frequency]),
    cell(r.key_metric), cell(r.baseline), cell(r.improvement, { bold: true })]);
  s.addTable([[hdr("ID"), hdr("Task"), hdr("Model"), hdr("Timeframe"), hdr("Key metric (test)"), hdr("Baseline"), hdr("Improvement [95% interval]")], ...rows],
    topts(L, 1.55, CW, [0.4, 1.05, 2.0, 0.95, 2.6, 2.75, CW - 9.75], { fontSize: 10, rowH: 0.37 }));
  const y = 1.55 + 0.37 + 0.45 * SEL.length + 0.05;
  const cw = (CW - 0.5) / 3;
  [["Strongest return approach", `${cfgName(bestRet)}: R² ${f2(bestRet.t_r2_oos_pct, 3)}%, ${sig(bestRet)}. Practically a random walk.`],
   ["Strongest direction approach", `${cfgName(bestDir)}: ${f2(bestDir.t_diff_pp)} pp over the majority class, ${sig(bestDir)}; Long & Short ${usd(trade("long_short").net_usd)} after $2.50 per round trip.`],
   ["Strongest volatility approach", `${VOL.map((r) => `${cfgName(r)}: R² ${volR2(r)} vs persistence, RV direction ${pctv(r.t_voldir_acc)} vs ${pctv(r.t_voldir_majority_acc)}`).join("; ")}.`]]
    .forEach(([h, b], i) => {
      const x = L + i * (cw + 0.25);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: cw, h: 1.0, fill: { color: i === 2 ? TINT : PALE }, line: { color: i === 2 ? RED : LINE, width: 1 } });
      T(s, [{ text: h, options: { bold: true, color: RED, breakLine: true } }, { text: b, options: { fontSize: 11 } }], { x: x + 0.12, y: y + 0.05, w: cw - 0.24, h: 0.92, fontSize: 12 });
    });
  const y2 = y + 1.12;
  T(s, [{ text: "Verdict: realized volatility is the most predictable characteristic. ", options: { bold: true, color: RED_DARK } },
        { text: "Its gain over the benchmark is large on both selected timeframes, with intervals far from zero. The return forecast does not beat zero, and the direction gain is under one percentage point and loses money after costs. Metrics differ by task, so the ranking rests on benchmark-relative gains and their uncertainty, not raw numbers. Test 2021–2025 is exploratory." }],
    { x: L, y: y2, w: CW, h: 0.8, fontSize: 11.5 });
  T(s, [{ text: "Originality ", options: { bold: true, color: RED } }, { text: "one pipeline for three targets, five timeframes, classical, deep and foundation models.   " },
        { text: "Usefulness ", options: { bold: true, color: RED } }, { text: "volatility forecasts for risk and sizing; no trading edge claimed.   " },
        { text: "Feasibility ", options: { bold: true, color: RED } }, { text: "runs on a laptop CPU, free local models, reproducible from the repository." }],
    { x: L, y: y2 + 0.8, w: CW, h: 0.5, fontSize: 11 });
}

pres.writeFile({ fileName: path.join(__dirname, OUT) }).then((f) => console.log("wrote " + f));
