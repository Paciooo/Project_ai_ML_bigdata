// NASDAQ multi-target deck (cover + 8 slides, statistical results only: no trading, P&L or costs).
// Every number comes from deck_data.json, a copy of reports/multitarget_final/deck_data.json
// written by multitarget_report.py (charts copied too).
// Build (Node.js 18+), from this folder: npm ci   (or npm install), then
//        node build_deck.js NASDAQ_multitarget_report.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || "NASDAQ_multitarget_report.pptx";
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const IMG = (f) => path.join(__dirname, f);
const REPO = "github.com/Paciooo/Project_ai_ML_bigdata";

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
const hdr = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, color: "FFFFFF", fill: { color: RED }, align: "center", valign: "middle" }, o) });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ valign: "middle" }, o) });
const num = (t, o = {}) => cell(t, Object.assign({ align: "center" }, o));
const topts = (x, y, w, colW, extra = {}) => Object.assign({ x, y, w, colW, fontFace: "Arial", fontSize: 11, color: INK,
  border: { type: "solid", pt: 0.75, color: LINE }, margin: [3, 5, 3, 5], rowH: 0.3 }, extra);

// ---------------------------------------------------------------- data helpers
const FREQ = { "5min": "5 min", "15min": "15 min", "30min": "30 min", "1h": "1 hour", session: "1 day" };
const FREQS = ["5min", "15min", "30min", "1h", "session"].filter((f) => D.dataset[f]);
const MODEL = { zero: "Zero return", train_mean: "Training mean", majority: "Majority class", persistence: "Persistence",
  rolling_mean_22: "Rolling mean (22)", har_ols: "HAR-OLS", ridge: "Ridge", logistic: "Logistic", random_forest: "Random forest",
  xgboost: "XGBoost", "mlp_64-32-16": "DNN 64-32-16", "mlp_64-32": "DNN 64-32", "mlp_128-64-32": "DNN 128-64-32",
  "mlp_256-128-64-32": "DNN 256-128-64-32", timestral: "Timestral-5M (frozen weights)",
  timestral_calibrated: "Timestral-5M + supervised vol. calibration" };
const TASK = { return: "Return", direction: "Direction", volatility: "Volatility" };
const SEL = D.selected;
const byTask = (t) => SEL.filter((r) => r.task === t);
const f2 = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const pctv = (v, d = 1) => (100 * v).toFixed(d) + "%";
const ciTxt = (lo, hi, d = 2, u = "") => `[${f2(lo, d)}${u}, ${f2(hi, d)}${u}]`;
const sig = (r) => r.ci_lo > 0 ? "interval above zero" : r.ci_hi < 0 ? "interval below zero" : "not distinguishable from zero";
const cfgName = (r) => `${MODEL[r.model]}, ${FREQ[r.frequency]}`;
const RET = byTask("return"), DIR = byTask("direction"), VOL = byTask("volatility");
const bestDir = DIR[0], bestRet = RET[0], bestVol = VOL[0];
const TS = D.timestral_outcome;
const volR2 = (r) => `${f2(r.t_r2_log_vs_persistence_pct, 0)}%`;
const volHar = (r) => `${f2(r.t_r2_log_vs_har_pct, 1)}%`;
const tsBeats = Object.entries(TS).filter(([f, v]) => v.vol_beats_persistence).map(([f]) => FREQ[f]);
const tsFrozenBeats = Object.values(TS).some((v) => v.return_beats_zero || v.direction_beats_majority);
const testRange = (f) => `${D.dataset[f].test[0]} → ${D.dataset[f].test[1]}`;
const PROF = (f, sp = "test") => D.data_profile.find((p) => p.frequency === f && p.split === sp);
const YEARS = (r) => D.direction_by_year[String(r.id)].filter((y) => y.year !== "all");
const kfmt = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k" : n.toLocaleString("en-US");
const range2 = (a) => { const v = a.map((x) => x.toFixed(2)); return v[0] === v[1] ? v[0] : `${v[0]}–${v[1]}`; };
const acfRange = (k) => { const v = FREQS.map((f) => PROF(f)[k]); return [Math.min(...v), Math.max(...v)]; };

// ================================================================ 1. Title (no links: repository on the last slide)
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

// ================================================================ 2. Executive summary (stands alone)
{
  const s = content("Executive Summary", "Volatility is clearly predictable; for returns and candle direction no useful improvement was demonstrated");
  // problem banner
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: L, y: 1.52, w: CW, h: 0.46, rectRadius: 0.08, fill: { color: PALE }, line: { type: "none" } });
  T(s, [{ text: "Problem  ", options: { bold: true, color: RED } },
        { text: "Markets are hard to forecast, but are all their characteristics equally hard? For the next NQ futures candle we ask what machine learning can predict: the return, the direction or the realized volatility." }],
    { x: L + 0.2, y: 1.52, w: CW - 0.4, h: 0.46, fontSize: 11.5, valign: "middle" });
  // three result cards
  const v1 = VOL[0], v2 = VOL[1] || VOL[0], y21 = YEARS(bestDir);
  const early = y21.filter((y) => y.ci_lo_pp > 0).map((y) => y.year), late = y21.filter((y) => y.ci_lo_pp <= 0).map((y) => y.year);
  const cards = [
    { task: "Volatility", badge: "PREDICTABLE", bc: GREEN, fill: RED, ink: "FFFFFF", sub: "F3D9DD",
      big: VOL.map((r) => `${f2(r.t_r2_log_vs_persistence_pct, 0)}%`).join("  /  "),
      cap: `less error than "next = current" (${VOL.map((r) => FREQ[r.frequency]).join(" / ")} candles)`,
      sec: `Also beats HAR-OLS (${VOL.map(volHar).join(" / ")}); rise or fall called right ${VOL.map((r) => pctv(r.t_voldir_acc, 0)).join(" / ")} (majority ${VOL.map((r) => pctv(r.t_voldir_majority_acc, 0)).join(" / ")})` },
    { task: "Direction", badge: "MARGINAL", bc: GOLD, fill: PALE, ink: INK, sub: MID,
      big: `${f2(bestDir.t_diff_pp)} pp`,
      cap: `accuracy over always-the-majority-class: ${pctv(bestDir.t_accuracy, 2)} vs ${pctv(bestDir.t_majority_acc, 2)} (${cfgName(bestDir)})`,
      sec: `About ${Math.round(bestDir.t_diff_pp * 10)} extra right calls per 1,000 candles; seen in ${early[0]}–${early[early.length - 1]}, ${late.length ? "gone in " + late.join("–") : "in every year"}` },
    { task: "Return", badge: "NOT DEMONSTRATED", bc: GREY, fill: PALE, ink: INK, sub: MID,
      big: `R² ${f2(bestRet.t_r2_oos_pct, 2)}%`,
      cap: `vs simply forecasting zero, 95% interval ${ciTxt(bestRet.ci_lo, bestRet.ci_hi, 2, "%")} (${cfgName(bestRet)})`,
      sec: `Forecasts move by less than ±0.5 bp while a typical ${FREQ[bestRet.frequency]} return is about ${bestRet.t_rmse_zero_bps.toFixed(1)} bp: the signal is lost in the noise` },
  ];
  const cw = (CW - 0.5) / 3, cy = 2.1, chh = 2.05;
  cards.forEach((c, i) => {
    const x = L + i * (cw + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: cy, w: cw, h: chh, rectRadius: 0.08, fill: { color: c.fill }, line: { color: i === 0 ? RED : LINE, width: 1 } });
    T(s, c.task, { x: x + 0.2, y: cy + 0.12, w: 1.7, h: 0.32, fontSize: 15, bold: true, color: i === 0 ? "FFFFFF" : RED, valign: "middle" });
    const bw = c.badge.length * 0.085 + 0.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + cw - bw - 0.18, y: cy + 0.15, w: bw, h: 0.27, rectRadius: 0.12, fill: { color: i === 0 ? "FFFFFF" : c.bc }, line: { type: "none" } });
    T(s, c.badge, { x: x + cw - bw - 0.18, y: cy + 0.15, w: bw, h: 0.27, fontSize: 9, bold: true, align: "center", valign: "middle", color: i === 0 ? GREEN : "FFFFFF" });
    T(s, c.big, { x: x + 0.2, y: cy + 0.48, w: cw - 0.4, h: 0.58, fontSize: 30, bold: true, color: c.ink, valign: "middle" });
    T(s, c.cap, { x: x + 0.2, y: cy + 1.06, w: cw - 0.4, h: 0.42, fontSize: 10.5, bold: true, color: c.ink });
    T(s, c.sec, { x: x + 0.2, y: cy + 1.5, w: cw - 0.4, h: 0.5, fontSize: 10, color: c.sub });
  });
  // why: autocorrelation chart
  const yb = 4.33, hb = 2.2, wc = 4.55;
  T(s, "Why: volatility clusters, returns do not", { x: L, y: yb - 0.02, w: wc, h: 0.3, fontSize: 12, bold: true, color: RED });
  s.addChart(pres.charts.BAR, [
    { name: "next return", labels: FREQS.map((f) => FREQ[f]), values: FREQS.map((f) => PROF(f).ret_acf1) },
    { name: "log realized variance", labels: FREQS.map((f) => FREQ[f]), values: FREQS.map((f) => PROF(f).lrv_acf1) }],
    { x: L - 0.05, y: yb + 0.28, w: wc + 0.1, h: hb - 0.3, barDir: "col", barGapWidthPct: 60, chartColors: [GREY, RED],
      valAxisMinVal: -0.2, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, showValue: true, dataLabelFormatCode: "0.00", dataLabelFontSize: 8, dataLabelPosition: "outEnd", valAxisLabelFontSize: 9, catAxisLabelFontSize: 9,
      valAxisTitle: "lag-1 autocorrelation (test)", showValAxisTitle: true, valAxisTitleFontSize: 9, valAxisLabelFormatCode: "0.0",
      showLegend: true, legendPos: "t", legendFontSize: 9, valGridLine: { color: "E6E6E6", size: 0.5 }, catAxisLineShow: true });
  // method / innovation / usefulness / conclusion
  const gx = L + wc + 0.3, gw = (CW - wc - 0.3 - 0.2) / 2, gh = (hb - 0.12) / 2;
  [["Method", `NQ futures 2006–2025, ${FREQS.length} candle lengths (5 min–1 day); ${D.n_models_tested} models, from benchmarks to deep networks and Mistral's Timestral. Train 2006–17, select 2017–21, test 2021–25.`],
   ["Innovation", "One pipeline, one data set and one test period for three different targets, so \"what is predictable\" is answered with the same models on every task and timeframe."],
   ["Usefulness", "Volatility forecasts feed risk limits, position sizing and volatility-sensitive decisions. Direction and return forecasts at these horizons are too weak to act on."],
   ["Conclusion", "Realized volatility is the most predictable characteristic; for returns and direction no useful improvement was demonstrated with the models, data and horizons tried."]]
    .forEach(([h, b], i) => {
      const x = gx + (i % 2) * (gw + 0.2), y = yb + Math.floor(i / 2) * (gh + 0.12);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: gw, h: gh, fill: { color: i === 3 ? TINT : PALE }, line: { type: "none" } });
      T(s, [{ text: h, options: { bold: true, color: RED, fontSize: 12, breakLine: true } }, { text: b, options: { fontSize: 10 } }],
        { x: x + 0.15, y: y + 0.08, w: gw - 0.3, h: gh - 0.12 });
    });
  T(s, "Exploratory: the test period (2021-07 → 2025-05; daily to 2025-01) was examined in earlier project phases and the volatility selection score was revised after a first look at test results. Intervals: 95% bootstrap in 20-session blocks. pp = percentage points, bp = 0.01%.",
    { x: L, y: 6.6, w: CW, h: 0.34, fontSize: 9, color: GREY });
}

// ================================================================ 3. Data & methodology
{
  const s = content("Data & Methodology", "One instrument, five candle lengths, three targets, one chronological split");
  const steps = ["NASDAQ historical data\nNQ 1-min, 2006–2025", "Feature engineering\nlags, momentum, volatility, time of day", "Models\nbenchmarks · ML · DNN · Timestral", "Forecasts\nfor candle t+1", `Out-of-sample evaluation\ntest from 2021-07 (exploratory)`];
  const bw = (CW - 4 * 0.3) / 5;
  steps.forEach((t, i) => {
    const x = L + i * (bw + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: bw, h: 0.85, rectRadius: 0.06, fill: { color: i === 4 ? RED : TINT }, line: { color: RED, width: 1 } });
    const [a, b] = t.split("\n");
    T(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b, options: { fontSize: 10 } }],
      { x: x + 0.08, y: 1.6, w: bw - 0.16, h: 0.85, fontSize: 11.5, align: "center", valign: "middle", color: i === 4 ? "FFFFFF" : INK });
    if (i < 4) s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + bw + 0.08, y: 1.93, w: 0.15, h: 0.2, rotate: 90, fill: { color: RED }, line: { type: "none" } });
  });
  header(s, L, 2.65, 6.1, "Three targets, all for the next candle (t+1), decided at the close of t");
  const tg = [
    ["1  Next return", "r = log(C(t+1)/C(t)), regression", "MAE, RMSE, out-of-sample R² vs zero forecast"],
    ["2  Next-candle direction", "UP if C(t+1) > O(t+1); doji kept, counts as a miss", "accuracy, balanced accuracy, F1 vs majority class"],
    ["3  Future realized volatility", "RV = Σ (1-min log returns)² inside candle t+1", "log-RV error (R²) vs persistence; RV UP if RV(t+1) > RV(t)"],
  ];
  s.addTable([[hdr("Target"), hdr("Definition"), hdr("Evaluated with")], ...tg.map(([a, b, c]) => [cell(a, { bold: true }), cell(b), cell(c)])],
    topts(L, 3.12, 6.1, [1.75, 2.3, 2.05], { fontSize: 10, rowH: 0.42 }));
  bullets(s, [
    [{ text: "Instrument: ", o: { bold: true } }, { text: "E-mini Nasdaq-100 futures (NQ, CME), back-adjusted continuous series. No QQQ, MNQ, cash index or VIX." }],
    [{ text: "Split: ", o: { bold: true } }, { text: `train ${D.dataset["5min"].train[0]} → ${D.dataset["5min"].train[1]}, validation → ${D.dataset["5min"].validation[1]}, test ${testRange("5min")} (daily test ends ${D.dataset.session.test[1]}).` }],
    [{ text: "Leakage control: ", o: { bold: true } }, { text: "features use candles ≤ t only; scaling fitted on training data; labels never cross a session or a split boundary." }],
    [{ text: "Selection: ", o: { bold: true } }, { text: "hyper-parameters and the five configurations chosen on validation; volatility score revised after a first look at test numbers, so not blind: results are exploratory." }],
  ], { x: L, y: 5.02, w: 6.1, h: 1.9, fontSize: 10.5 });
  // data profile
  const x2 = L + 6.45, w2 = CW - 6.45;
  header(s, x2, 2.65, w2, "Data profile per timeframe (test sample)");
  const rows = FREQS.map((f) => { const d = D.dataset[f], p = PROF(f);
    return [num(FREQ[f], { bold: true }), num(`${kfmt(d.train[2])} · ${kfmt(d.validation[2])} · ${kfmt(d.test[2])}`),
      num(`${pctv(p.share_up, 0)} / ${pctv(p.share_down, 0)} / ${pctv(p.share_doji, 1)}`), num(p.ret_sd_bps.toFixed(1)),
      num(f2(p.ret_acf1, 3), { color: MID }), num(f2(p.lrv_acf1), { bold: true, color: RED })]; });
  s.addTable([[hdr("Candle"), hdr("Rows: train · val · test"), hdr("UP / DOWN / doji"), hdr("Return s.d. (bp)"), hdr("ACF return"), hdr("ACF log RV")], ...rows],
    topts(x2, 3.12, w2, [0.8, 1.55, 1.25, 0.72, 0.73, 0.73], { fontSize: 10, rowH: 0.36 }));
  const [ra, rb] = acfRange("ret_acf1"), [va, vb] = acfRange("lrv_acf1");
  T(s, [{ text: "Reading: ", options: { bold: true } },
        { text: `candle direction is close to a coin flip (UP ${pctv(Math.min(...FREQS.map((f) => PROF(f).share_up)), 0)}–${pctv(Math.max(...FREQS.map((f) => PROF(f).share_up)), 0)}); ACF (lag-1 autocorrelation; intraday pairs within a session) of returns is about zero (${f2(ra, 3)} to ${f2(rb, 3)}), while log realized variance is strongly persistent (${va.toFixed(2)} to ${vb.toFixed(2)}). ` },
        { text: `Features (${D.dataset[FREQS[0]].features.length}): past returns, momentum, rolling volatility, range, body, wicks, RSI, moving-average gaps, relative volume, time of day, past log RV (1, 5, 22 candles, same slot on 5 prior sessions). Timestral sees the last 512 closes.`, options: { color: MID } }],
    { x: x2, y: 5.42, w: w2, h: 1.5, fontSize: 9.5 });
}

// ================================================================ 4. All models tested
{
  const s = content("All Models Tested", `${D.n_models_tested} models on ${FREQS.length} timeframes: ${D.n_configs} model–timeframe–task configurations, all run on the same data`);
  const fam = [
    ["Baselines", [["Zero return / training mean", "naive forecast", "R"], ["Majority class", "always the most frequent training class", "D"], ["Persistence", "RV(t+1) = RV(t)", "V"], ["Rolling mean (22 candles)", "average past variance", "V"]]],
    ["Statistical", [["HAR-OLS", "linear regression on past log RV (1, 5, 22 candles, same time slot)", "V"], ["Ridge regression", "L2-penalised linear, α chosen on validation", "R · V"], ["Logistic regression", "L2-penalised, C chosen on validation", "D"]]],
    ["Machine learning", [["Random forest", "200 trees, depth 8", "R · D · V"], ["XGBoost", "gradient boosting, depth 4, early stopping", "R · D · V"]]],
    ["Deep learning", [["DNN 64-32-16 (original)", "feed-forward, ReLU, AdamW, early stopping", "R · D · V"], ["DNN 64-32", "", "R · D · V"], ["DNN 128-64-32", "", "R · D · V"], ["DNN 256-128-64-32", "", "R · D · V"]]],
    ["Mistral AI", [["Timestral-5M, frozen weights", "zero-shot quantiles of the next close from 512 past closes; median → return; direction = median vs last close (target: next candle's body)", "R · D"],
                    ["Timestral-5M + supervised calibration", "q10–q90 spread → log RV via a linear map fitted on validation targets (on validation: fitted on earlier sessions only)", "V"]]],
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
    [{ text: "Timestral outcome: ", o: { bold: true } }, { text: `frozen weights ${tsFrozenBeats ? "beat a benchmark on some timeframe" : "beat no benchmark on any timeframe (return, direction)"}; the calibrated volatility variant beat persistence only at ${tsBeats.join(", ") || "no timeframe"}. Neither is among the five; per-timeframe numbers in the report tables.` }],
    [{ text: "Considered, not run: ", o: { bold: true } }, { text: "LSTM (the feature networks reach their early-stopping optimum within a few epochs). Timestral 28M / 123M / 209M: weights available locally, not tested in this study." }],
    [{ text: "Not used: ", o: { bold: true } }, { text: "Mistral chat / LLM API — no such integration exists in the repository." }],
  ], { x: x2, y: 1.98, w: w2, h: 4.9, fontSize: 10.5 });
}

// ================================================================ 5. Return forecasting
{
  const s = content("Price / Return Forecasting", RET.every((r) => r.ci_lo <= 0) ? "No improvement over the zero-return forecast demonstrated with the models, data and horizons tried"
    : "A small improvement over the zero-return forecast (exploratory)");
  const rows = RET.map((r) => [num(String(r.id), { bold: true }), cell(cfgName(r), { bold: true }), num(r.t_mae_bps.toFixed(2)), num(r.t_rmse_bps.toFixed(2)),
    num(r.t_rmse_zero_bps.toFixed(2)), num(`${f2(r.t_r2_oos_pct, 3)}%`, { bold: true, color: r.ci_lo > 0 ? GREEN : INK }), num(ciTxt(r.ci_lo, r.ci_hi, 3, "%")), num(pctv(r.t_dir_acc))]);
  s.addTable([[hdr("#"), hdr("Model, timeframe"), hdr("MAE (bp)"), hdr("RMSE (bp)"), hdr("RMSE zero (bp)"), hdr("R² vs zero"), hdr("95% interval"), hdr("Sign right")], ...rows],
    topts(L, 1.6, 6.6, [0.3, 1.9, 0.6, 0.65, 0.8, 0.75, 1.0, 0.6], { fontSize: 10, rowH: 0.4 }));
  s.addImage({ path: IMG(D.charts[String(bestRet.id)]), x: L + 6.9, y: 1.55, w: CW - 6.9, h: (CW - 6.9) * 3.6 / 5.6 });
  const n = RET.map((r) => `${cfgName(r)}: R² ${f2(r.t_r2_oos_pct, 3)}%, interval ${sig(r)}`).join("; ");
  bullets(s, [
    [{ text: "Result: ", o: { bold: true } }, { text: `${n}. A positive R² would mean a smaller squared error than always forecasting zero.` }],
    [{ text: "Predicted vs realized (right): ", o: { bold: true } }, { text: `realized returns per forecast decile. The forecasts span less than ±0.5 bp, while a typical candle moves about ${bestRet.t_rmse_zero_bps.toFixed(1)} bp (RMSE): the signal is buried in noise.` }],
    [{ text: "Economic reading: ", o: { bold: true } }, { text: "no usable return signal was found with the models, data and horizons tried. This is consistent with near-random-walk returns at these horizons, but does not prove it. All return models and Timestral were run on all five timeframes (inventory, slide 4)." }],
  ], { x: L, y: 2.75, w: 6.6, h: 3.6, fontSize: 11.5 });
  T(s, `bp = basis point = 0.01%. R² interval: model and zero-forecast errors resampled together in 20-session blocks, ratio recomputed per replicate. Decile bars: 20-session block bootstrap within each decile. Test ${testRange(bestRet.frequency)} (exploratory: examined in earlier project phases).`,
    { x: L + 6.9, y: 5.55, w: CW - 6.9, h: 0.9, fontSize: 9.5, color: GREY });
}


// ================================================================ 6. Direction classification
{
  const c = bestDir, yrs = YEARS(c);
  const pos = yrs.filter((y) => y.ci_lo_pp > 0).map((y) => y.year), zero = yrs.filter((y) => y.ci_lo_pp <= 0 && y.ci_hi_pp >= 0).map((y) => y.year);
  const s = content("Next-Candle Direction Classification", `A small gain over the majority class (${f2(c.t_diff_pp)} pp, ${sig(c)}) that is not stable across test years`);
  const rows = DIR.map((r) => [num(String(r.id), { bold: true }), cell(cfgName(r), { bold: true }), num(pctv(r.t_accuracy, 2), { bold: true }),
    num(pctv(r.t_balanced_acc, 2)), num(pctv(r.t_majority_acc, 2)), num(`${f2(r.t_diff_pp)} pp`, { bold: true, color: r.ci_lo > 0 ? GREEN : INK }), num(ciTxt(r.ci_lo, r.ci_hi)),
    num(pctv(r.t_precision_up)), num(pctv(r.t_recall_up)), num(r.t_f1_up.toFixed(3))]);
  s.addTable([[hdr("#"), hdr("Model, timeframe"), hdr("Accuracy"), hdr("Balanced acc."), hdr("Majority"), hdr("Gain"), hdr("95% interval"),
               hdr("Precision UP"), hdr("Recall UP"), hdr("F1 UP")], ...rows],
    topts(L, 1.55, CW, [0.3, 2.3, 1.05, 1.15, 1.0, 1.0, 1.5, 1.25, 1.15, CW - 10.7], { fontSize: 10.5, rowH: 0.4 }));
  // confusion matrix of the best configuration
  const ty = 1.55 + 0.4 * (DIR.length + 1) + 0.3;
  header(s, L, ty, 5.4, `Confusion matrix, ${cfgName(c)} (test)`);
  const n = c.t_tp + c.t_fp + c.t_tn + c.t_fn, sh = (v) => `${v.toLocaleString("en-US")}\n${pctv(v / n)}`;
  s.addTable([[cell(""), hdr("Actual UP"), hdr("Actual DOWN")],
              [hdr("Predicted UP"), num(sh(c.t_tp), { bold: true, fill: { color: TINT } }), num(sh(c.t_fp))],
              [hdr("Predicted DOWN"), num(sh(c.t_fn)), num(sh(c.t_tn), { bold: true, fill: { color: TINT } })]],
    topts(L, ty + 0.48, 5.4, [1.6, 1.9, 1.9], { fontSize: 10.5, rowH: 0.5 }));
  T(s, `Non-doji test candles. Dojis (open = close), counted as misses: ${c.t_n_doji.toLocaleString("en-US")}. Predicted UP on ${pctv(c.t_pred_up_share)} of candles.`,
    { x: L, y: ty + 2.05, w: 5.4, h: 0.3, fontSize: 9.5, color: MID });
  bullets(s, [
    [{ text: "Size: ", o: { bold: true } }, { text: `${f2(c.t_diff_pp)} pp is about ${Math.round(c.t_diff_pp * 10)} more right calls per 1,000 candles; balanced accuracy ${pctv(c.t_balanced_acc, 2)}.` }],
  ], { x: L, y: ty + 2.6, w: 5.4, h: 0.6, fontSize: 11 });
  // per-year chart
  const cx = L + 5.75, cw2 = CW - 5.75, ch = cw2 * 3.6 / 6.4;
  s.addImage({ path: IMG(D.charts[String(c.id)]), x: cx, y: ty - 0.05, w: cw2, h: ch > 6.9 - ty - 0.85 ? 6.9 - ty - 0.85 : ch, sizing: { type: "contain", w: cw2, h: Math.min(ch, 6.9 - ty - 0.85) } });
  bullets(s, [
    [{ text: "Stability (right): ", o: { bold: true } }, { text: `gain above zero in ${pos.join(", ") || "no year"}${zero.length ? `; interval includes zero in ${zero.join(", ")}` : ""}. The overall gain comes from the early test years.` }],
  ], { x: cx, y: 6.9 - 0.8, w: cw2, h: 0.75, fontSize: 11 });
}

// ================================================================ 7. Volatility
{
  const s = content("Volatility Forecasting", "Large gain over persistence, smaller but positive gain over the HAR-OLS benchmark (exploratory)");
  const ci1 = (v, lo, hi, d = 1, u = "%") => `${f2(v, d)}${u}\n${ciTxt(lo, hi, d)}`;
  const rows = VOL.map((r) => [num(String(r.id), { bold: true }), cell(cfgName(r), { bold: true }), num(r.t_rmse_log.toFixed(3), { bold: true }),
    num(r.t_rmse_log_persistence.toFixed(3)), num(r.har_rmse_log.toFixed(3)),
    num(ci1(r.t_r2_log_vs_persistence_pct, r.ci_lo, r.ci_hi), { color: GREEN, bold: true }),
    num(ci1(r.t_r2_log_vs_har_pct, r.t_r2_log_vs_har_ci_lo_pct, r.t_r2_log_vs_har_ci_hi_pct), { color: r.t_r2_log_vs_har_ci_lo_pct > 0 ? GREEN : INK, bold: true }),
    num(pctv(r.t_voldir_acc), { bold: true }), num(pctv(r.t_voldir_balanced_acc)), num(pctv(r.t_voldir_majority_acc)), num(pctv(r.t_voldir_acc_har)),
    num(ci1(r.t_voldir_diff_vs_har_pp, r.t_voldir_vs_har_ci_lo_pp, r.t_voldir_vs_har_ci_hi_pp, 2, " pp"), { color: r.t_voldir_vs_har_ci_lo_pp > 0 ? GREEN : INK, bold: true })]);
  const cwT = [0.3, 1.75, 0.8, 0.75, 0.75, 1.2, 1.2, 0.85, 0.85, 0.85, 0.85];
  cwT.push(CW - cwT.reduce((a, b) => a + b, 0));
  s.addTable([[hdr("#"), hdr("Model, timeframe"), hdr("RMSE log RV"), hdr("Persist."), hdr("HAR-OLS"), hdr("R² vs persistence"), hdr("R² vs HAR-OLS"),
               hdr("RV dir. acc."), hdr("Balanced"), hdr("Majority"), hdr("HAR dir. acc."), hdr("Dir. gain vs HAR")], ...rows],
    topts(L, 1.55, CW, cwT, { fontSize: 9.5, rowH: 0.42 }));
  const yb = 1.55 + 0.55 + 0.62 * VOL.length + 0.15;
  const ch = (CW - 7.6) * 3.6 / 5.6;
  s.addImage({ path: IMG(D.charts[String(bestVol.id)]), x: L + 7.6, y: yb - 0.05, w: CW - 7.6, h: ch });
  bullets(s, [
    [{ text: "Method: ", o: { bold: true } }, { text: `realized variance of the next candle from its 1-minute squared log returns (r² is a variance proxy; volatility = √variance). Models forecast log RV; R² = share of the benchmark's squared log-RV error removed. QLIKE check, model / HAR / persistence: ${VOL.map((r) => `${r.t_qlike.toFixed(3)} / ${r.har_qlike.toFixed(3)} / ${r.t_qlike_persistence.toFixed(3)}`).join("; ")}.` }],
    [{ text: "Two benchmarks: ", o: { bold: true } }, { text: `persistence is naive; HAR-OLS already captures clustering, mean reversion and the time-of-day pattern (RV direction ${VOL.map((r) => pctv(r.t_voldir_acc_har)).join(" / ")}). The gain vs HAR, not vs persistence, is what ML adds.` }],
    [{ text: "Options (qualitative): ", o: { bold: true } }, { text: "rising realized volatility favours long straddles / strangles, falling favours short ones; but option prices embed IMPLIED volatility, so a correct realized-volatility forecast is not a profitable options trade. No option prices or P&L were computed." }],
  ], { x: L, y: yb, w: 7.3, h: 6.45 - yb, fontSize: 10.5 });
  T(s, "Selection score: initially QLIKE vs persistence; revised to log-RV squared error after first test numbers were printed (QLIKE was dominated by a few limit-locked March 2020 candles): not blind. R² intervals: model and benchmark errors resampled together in 20-session blocks; direction gains: paired block bootstrap.",
    { x: L, y: 6.42, w: CW, h: 0.5, fontSize: 9, color: GREY });
}

// ================================================================ 8. Final comparison
{
  const s = content("Which NASDAQ Market Characteristic Is Most Predictable?", "Five configurations chosen with validation data (volatility score revised after a first look at test results), scored on the exploratory test");
  const impr = (r) => r.task !== "volatility" ? r.improvement
    : `vs persistence R² ${f2(r.t_r2_log_vs_persistence_pct, 1)}% ${ciTxt(r.ci_lo, r.ci_hi, 1)}; vs HAR R² ${f2(r.t_r2_log_vs_har_pct, 1)}% ${ciTxt(r.t_r2_log_vs_har_ci_lo_pct, r.t_r2_log_vs_har_ci_hi_pct, 1)}, RV dir. ${f2(r.t_voldir_diff_vs_har_pp)} pp`;
  const base = (r) => r.task !== "volatility" ? r.baseline : `${r.baseline}; HAR ${r.har_rmse_log.toFixed(3)}, ${pctv(r.t_voldir_acc_har)}`;
  const rows = SEL.map((r) => [num(String(r.id), { bold: true }), cell(TASK[r.task], { bold: true, color: RED }), cell(MODEL[r.model]), num(FREQ[r.frequency]),
    cell(r.key_metric), cell(base(r)), cell(impr(r), { bold: true })]);
  s.addTable([[hdr("ID"), hdr("Task"), hdr("Model"), hdr("Timeframe"), hdr("Key metric (test)"), hdr("Baseline"), hdr("Improvement [95% interval]")], ...rows],
    topts(L, 1.55, CW, [0.35, 0.95, 1.75, 0.85, 2.35, 2.75, CW - 9.0], { fontSize: 9.5, rowH: 0.37 }));
  const y = 1.55 + 0.37 + 0.4 * 3 + 0.58 * 2 + 0.1;
  const cw = (CW - 0.5) / 3;
  const yrs = YEARS(bestDir), late = yrs.filter((x) => x.ci_lo_pp <= 0).map((x) => x.year);
  [["Strongest return approach", `${cfgName(bestRet)}: R² ${f2(bestRet.t_r2_oos_pct, 3)}%, ${sig(bestRet)}: no improvement over zero demonstrated with the models, data and horizons tried.`],
   ["Strongest direction approach", `${cfgName(bestDir)}: ${f2(bestDir.t_diff_pp)} pp over the majority class, ${sig(bestDir)}; ${late.length ? `not distinguishable from zero in ${late.join(", ")}` : "positive in every test year"}.`],
   ["Strongest volatility approach", `log-RV R² vs persistence / vs HAR-OLS: ${VOL.map((r) => `${cfgName(r)} ${volR2(r)} / ${volHar(r)}`).join("; ")}. RV direction above HAR by ${VOL.map((r) => `${f2(r.t_voldir_diff_vs_har_pp)} pp`).join(" / ")}.`]]
    .forEach(([h, b], i) => {
      const x = L + i * (cw + 0.25);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: cw, h: 1.1, fill: { color: i === 2 ? TINT : PALE }, line: { color: i === 2 ? RED : LINE, width: 1 } });
      T(s, [{ text: h, options: { bold: true, color: RED, breakLine: true } }, { text: b, options: { fontSize: 10.5 } }], { x: x + 0.12, y: y + 0.05, w: cw - 0.24, h: 1.0, fontSize: 12 });
    });
  const y2 = y + 1.2;
  T(s, [{ text: "Exploratory verdict: realized volatility is the most predictable characteristic. ", options: { bold: true, color: RED_DARK } },
        { text: "Its gain is large over persistence and smaller but positive over HAR-OLS, with intervals above zero on both selected timeframes, consistent with the strong persistence of log realized variance in the data. For returns no improvement over zero was demonstrated; the direction gain is under one percentage point and not stable over time. Metrics differ by task, so the ranking rests on benchmark-relative gains and their uncertainty." }],
    { x: L, y: y2, w: CW, h: 0.8, fontSize: 11.5 });
  T(s, [{ text: "Originality ", options: { bold: true, color: RED } }, { text: "one pipeline for three targets, five timeframes, classical, deep and foundation models.   " },
        { text: "Usefulness ", options: { bold: true, color: RED } }, { text: "volatility forecasts for risk limits and sizing.   " },
        { text: "Feasibility ", options: { bold: true, color: RED } }, { text: "runs on a laptop CPU with free local models; reproducible (next slide)." }],
    { x: L, y: y2 + 0.8, w: CW, h: 0.5, fontSize: 11 });
}

// ================================================================ 9. Reproducibility
{
  const s = content("Reproducibility", "All tables, charts and slides are regenerated from the code in the repository; the licensed NQ data are not published");
  // repository card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: L, y: 1.6, w: CW, h: 0.8, rectRadius: 0.08, fill: { color: RED }, line: { type: "none" } });
  T(s, "Code, results and this deck", { x: L + 0.3, y: 1.6, w: 3.2, h: 0.8, fontSize: 13, bold: true, color: "F3D9DD", valign: "middle" });
  T(s, [{ text: `https://${REPO}`, options: { hyperlink: { url: `https://${REPO}` }, color: "FFFFFF", bold: true } },
        { text: "   ·   branch main", options: { color: "F3D9DD" } }],
    { x: L + 3.4, y: 1.6, w: CW - 3.7, h: 0.8, fontSize: 18, valign: "middle" });
  // steps
  const steps = [
    ["1  Environment", "Python 3.12, Node.js 18+", "pip install -r requirements.txt -r requirements-timestral.txt"],
    ["2  Data (not in Git)", "NQ 1-minute bars 2006–2025 (TradeStation, licensed); Timestral-5M weights from the local Hugging Face cache", "$env:FUTURES_DATA_DIR = \"C:\\path\\to\\futures-data\""],
    ["3  Train and forecast", "all models, five timeframes, CPU only", "python multitarget.py --tag final"],
    ["4  Tables and charts", "results_*.csv, data_profile.csv, selection, charts", "python multitarget_report.py <run> reports\\multitarget_final"],
    ["5  Independent check", "recomputes every number from the saved forecasts", "python verify_multitarget.py <run> reports\\multitarget_final presentation"],
    ["6  Tests and deck", "unit tests; slides from deck_data.json", "python -m pytest -q   ·   cd presentation; npm ci; node build_deck.js"],
  ];
  const body = [[hdr("Step"), hdr("What"), hdr("Command (PowerShell, repository root)")],
    ...steps.map(([a, b, c]) => [cell(a, { bold: true, color: RED }), cell(b, { color: MID }), cell(c, { fontFace: "Consolas", fontSize: 9.5 })])];
  s.addTable(body, topts(L, 2.62, 8.5, [1.75, 2.85, 3.9], { fontSize: 10, rowH: 0.5 }));
  // checks and limits
  const x2 = L + 8.8, w2 = CW - 8.8;
  header(s, x2, 2.55, w2, "What has been checked");
  bullets(s, [
    `verify_multitarget.py: every metric, the selection rule, the data profile and the deck data recomputed independently; 0 problems on the final run`,
    "unit tests for targets, bootstrap, selection rule and the causal Timestral calibration",
    `run ${D.run_id}`,
  ], { x: x2, y: 3.0, w: w2, h: 1.85, fontSize: 10.5 });
  header(s, x2, 4.85, w2, "Limits");
  bullets(s, [
    "the test period had been examined before: results are exploratory",
    "without the licensed NQ data the training cannot be re-run; the saved forecasts are not published either",
    "the run was made from a working copy; run.json stores the hash of every source file",
  ], { x: x2, y: 5.3, w: w2, h: 1.6, fontSize: 10.5 });
}

pres.writeFile({ fileName: path.join(__dirname, OUT) }).then((f) => console.log("wrote " + f));
