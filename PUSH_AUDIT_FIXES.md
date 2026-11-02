# Push manuale: correzioni e pulizia

Esegui dalla cartella del progetto, un comando alla volta. Fermati in caso di
errore. Nessun commit o push automatico e stato fatto durante la pulizia.

Il primo blocco allinea l'indice solo se contiene ancora vecchie aggiunte
temporanee di `tmp` e `logs`, gia eliminate dal disco. Non modifica i dati
grezzi o config.py.

```powershell
$obsoleteIndexed = @(git ls-files -- tmp logs)
if ($obsoleteIndexed.Count -gt 0) { git restore --staged -- tmp logs }
```

```powershell
$fixPaths = @(
    "backtest.py", "experiment.py", "features.py", "plots.py",
    "tools/repair_results.py", "tools/coinflip_vix.py",
    "tests/test_execution_repairs.py", "tests/test_models_and_backtest.py",
    ".gitignore", "README.md", "CLAUDE.md", "RUN_GUIDE.md", "PUSH_AUDIT_FIXES.md",
    "FINAL_RESULTS_SUMMARY.md", "MONTE_CARLO_VIX_RESULTS.md",
    "PROJECT_REVIEW.md", "CLAUDE_CODE_FIX_PROMPT.md", "docs", "reports"
)
```

I nomi Markdown non piu presenti nella radice servono a registrare in Git il
loro spostamento dentro `docs`. La cartella `reports` include anche la
rimozione del vecchio run cloud.

```powershell
git add -A -- $fixPaths
```

```powershell
git commit --only -m "Fix backtests, verify results and simplify project handoff" -- $fixPaths
```

```powershell
git fetch origin
```

```powershell
git push origin HEAD:main
```

L'ultimo comando pubblica su `main` i commit del branch corrente. Le altre
modifiche eventualmente gia presenti nell'indice, come config.py e il PDF,
restano fuori da questo commit selettivo.
