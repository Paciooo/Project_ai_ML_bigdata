import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("merge_daily_run", Path(__file__).resolve().parents[1] / "tools" / "merge_daily_run.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def _run(d, freqs, tag):
    d.mkdir()
    for f in freqs:
        pd.DataFrame({"x": [tag]}).to_parquet(d / f"predictions_{f}.parquet")
    pd.DataFrame({"frequency": list(freqs), "model": tag}).to_csv(d / "models.csv", index=False)
    (d / "dataset_info.json").write_text(json.dumps({f: {"src": tag} for f in freqs}))
    (d / "run.json").write_text(json.dumps({"tag": tag}))
    (d / "console.log").write_text(tag)


def test_merge_takes_daily_from_daily_run_and_keeps_inputs(tmp_path):
    base, daily = tmp_path / "base", tmp_path / "daily"
    _run(base, ["5min", "session"], "base")
    _run(daily, ["session"], "daily")
    merge.main(base, daily, "merged")
    out = tmp_path / "daily_merged"
    assert pd.read_parquet(out / "predictions_5min.parquet").x[0] == "base"
    assert pd.read_parquet(out / "predictions_session.parquet").x[0] == "daily"
    info = json.loads((out / "dataset_info.json").read_text())
    assert info == {"5min": {"src": "base"}, "session": {"src": "daily"}}
    m = pd.read_csv(out / "models.csv")
    assert set(zip(m.frequency, m.model)) == {("5min", "base"), ("session", "daily")}
    assert pd.read_parquet(base / "predictions_session.parquet").x[0] == "base"     # inputs untouched
    with pytest.raises(SystemExit):                                                  # never overwrites
        merge.main(base, daily, "merged")


def test_merge_rejects_daily_run_with_intraday(tmp_path):
    base, daily = tmp_path / "base", tmp_path / "daily"
    _run(base, ["5min", "session"], "base")
    _run(daily, ["5min", "session"], "daily")
    with pytest.raises(SystemExit):
        merge.main(base, daily, "merged")
