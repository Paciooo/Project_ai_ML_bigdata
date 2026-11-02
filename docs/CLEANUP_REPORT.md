# Pulizia e contesto per Claude

Eseguita il 2 novembre 2026 su richiesta esplicita dell'utente.

## Cosa aprire adesso

1. `README.md`: mappa sintetica del progetto.
2. `CLAUDE.md`: contesto corrente per Claude Code, correzioni e risultati.
3. `reports/README.md`: indice dei risultati verificati da usare.
4. `docs/`: analisi dettagliate; `docs/history/` contiene solo materiale storico.

## Cosa e stato eliminato

- Due run VIX intermedi/incompleti, gia sostituiti dal run con tutti i modelli.
- Output degli smoke test e relativi modelli casuali di prova.
- Il riepilogo del vecchio run A cloud, che non va mescolato con A locale.
  La vecchia versione rimane nella cronologia Git.
- Vecchi CSV/grafici NQ precedenti alla nuova pipeline e tabelle aggregate
  ridondanti rispetto ai run completi.
- Copie ridondanti degli output sintetici, log precedenti alle correzioni,
  file temporanei, immagini estratte dal PDF e cache rigenerabili.
- Un backup incompleto di un tentativo fallito. I backup della correzione
  completata, citati nei manifest di verifica, sono stati conservati.

Totale: **139 file rimossi**, circa **4,76 MiB**. Lo scopo era eliminare
duplicazioni e ambiguita, non cancellare i dati utili per recuperare spazio.
L'elenco dei 34 percorsi rimossi e in
`reports/verification/cleanup_manifest.json`.

## Cosa e stato conservato

- Cinque run completi: A, B, C, Timestral-NQ e VIX/Monte Carlo definitivo.
- Previsioni complete, modelli salvati e una copia precedente alle correzioni.
- Tutti i report piccoli aggiornati, grafici e prove della verifica locale.
- Un unico riepilogo del controllo GBM in `reports/synthetic_null_legacy/`.
- Dati IBKR, dati grezzi NQ/VIX, pesi Timestral, PDF originale, ambiente
  Python, configurazione utente e cronologia Git.

I dati NQ/VIX esterni al progetto non sono stati spostati o modificati.
I file di addestramento e le probabilita salvate non sono cambiati.

## Documenti riorganizzati

Le analisi attuali sono in `docs/AUDIT_FIXES.md`,
`docs/FINAL_RESULTS_SUMMARY.md` e `docs/MONTE_CARLO_VIX_RESULTS.md`.
La prima revisione, il vecchio prompt di correzione e il README esteso sono
in `docs/history/`, etichettati come archivio. Non sono nuove cose da fare.

`run.json` conserva la provenienza originale. Alcuni suoi riferimenti a run
intermedi descrivono la storia dell'esecuzione; non sono dipendenze correnti.
Tutti i risultati riutilizzati sono presenti nel run definitivo mantenuto.

La pulizia non richiede nuovi esperimenti. Dopo la riorganizzazione sono
passati nuovamente **70 test**. Verificati anche: esistenza dei cinque run,
integrita dei backup citati, hash invariati dei dati grezzi NQ/VIX e validita
dei collegamenti Markdown correnti. I test sono stati eseguiti senza
ricreare cache e cartelle temporanee nella radice del progetto.
