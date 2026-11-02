# Previsione della direzione: NQ e futures VIX

Progetto universitario. L'analisi e completa ed e dimensionata per una
presentazione di **massimo 10 slide**. Non e necessario rilanciare i run.

## Da dove iniziare

| Ti serve... | Apri |
|---|---|
| Contesto aggiornato per Claude Code | [CLAUDE.md](CLAUDE.md) |
| Risultati da usare nelle slide | [Indice dei report](reports/README.md) |
| Sintesi NQ, A/B/C e Timestral | [Riepilogo NQ](docs/FINAL_RESULTS_SUMMARY.md) |
| Moneta casuale e VIX, tutti i Timestral | [Riepilogo Monte Carlo/VIX](docs/MONTE_CARLO_VIX_RESULTS.md) |
| Correzioni e verifiche effettuate | [Audit finale](docs/AUDIT_FIXES.md) |
| Comandi tecnici, solo se servono | [RUN_GUIDE.md](RUN_GUIDE.md) |
| Pubblicazione manuale su GitHub | [PUSH_AUDIT_FIXES.md](PUSH_AUDIT_FIXES.md) |

## Risultato principale

- NQ: piccolo vantaggio statistico a 1 e 5 minuti; nessun miglioramento
  dimostrato sulle candele piu lunghe con le feature/modelli provati.
- VIX: si prevede la direzione del future con **scadenza piu vicina**,
  mantenendo la stessa scadenza nel contesto storico ed escludendo i cambi
  contratto dal target. Non e una previsione della volatilita realizzata SPX.
- VIX majority: 59,59%; logistica: 59,81%, senza miglioramento dimostrato.
  Timestral 5M, 28M, 123M e 209M sono stati eseguiti sugli stessi 928 casi.
- La moneta equa non basta come confronto: serve anche la majority e una
  moneta con la stessa quota attesa di UP del modello.
- Profitto non dimostrato. A 1 minuto, con latenza positiva e soli prezzi
  minuto, l'ingresso prima della fine del target non e osservabile: P&L N/A.
  Sugli altri orizzonti restano simulazioni con prezzi proxy e costi ipotizzati.

## Cartelle: cosa conservano

```text
CLAUDE.md          contesto corrente per l'agente
docs/              analisi attuale e audit
  history/         vecchie revisioni/prompt, solo archivio
reports/           riepiloghi piccoli verificati, grafici e provenienza
results/           solo i 5 run completi locali e relativi backup utili
data_raw/          download IBKR e cache dei pesi: non eliminare
tests/             70 test superati nell'ultima verifica
tools/             download, Timestral, Monte Carlo e verifica/ricalcolo
intresting paper/  PDF originale di riferimento
```

`reports/` e la cartella da usare per consultare o condividere i risultati.
`results/` conserva i file completi necessari a verificare/ricalcolare.
Non mescolare i dati di run diversi. I vecchi run intermedi, smoke e output
legacy duplicati sono stati rimossi. Del controllo GBM rimane un solo
riepilogo, chiaramente etichettato `reports/synthetic_null_legacy/`.

## Dati locali

NQ storico: `C:\Users\aless\Desktop\py\futures-data`.
VIX: CSV originali nella cartella `Scuola\BSAMC\VIX\Codice\data`.
Non serve spostarli nel progetto. I dati grezzi, i pesi, `.venv` e `.git`
non sono stati eliminati durante la pulizia.

## Controllo rapido

```powershell
python -m pytest tests -q
```

I run scientifici gia completati non vanno ripetuti per preparare le slide.
Le istruzioni precedenti e le tabelle storiche sono in `docs/history/` e non
descrivono lo stato corrente. I dettagli della pulizia sono in
[docs/CLEANUP_REPORT.md](docs/CLEANUP_REPORT.md).
