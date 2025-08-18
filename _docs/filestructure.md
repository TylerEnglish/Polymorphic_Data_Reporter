```
polymorphic-data-reporter/
├─ config/
│  └─ config.toml
├─ data/
│  ├─ raw/
│  ├─ bronze/
│  ├─ silver/
│  └─ gold/
│     └─ <dataset_slug>/
│        ├─ tables/
│        ├─ artifacts/
│        ├─ reports/
│        └─ manifest.json
├─ docker/
│  ├─ docker-compose.yml
│  ├─ airflow.Dockerfile
│  ├─ chrome.Dockerfile
│  └─ env/
│     ├─ airflow.env
│     └─ minio.env
├─ airflow/
│  ├─ dags/
│  │  └─ polymorphic_report_dag.py
│  └─ include/
├─ schemas/
│  └─ <dataset_slug>.schema.toml
├─ src/
│  ├─ config_model/
│  │  ├─ __init__.py
│  │  └─ model.py
│  ├─ io/
│  │  ├─ storage.py          # local + MinIO S3 abstraction
│  │  ├─ readers.py          # csv/json/parquet/duckdb
│  │  ├─ writers.py          # csv/json/parquet/html/png/pdf
│  │  └─ catalog.py
│  ├─ ingestion/
│  │  └─ ingest.py
│  ├─ profiling/
│  │  ├─ roles.py
│  │  ├─ stats.py
│  │  └─ profile.py
│  ├─ cleaning/              # <<< new: all cleaning rulepacks + engine
│  │  ├─ engine.py           # compiles + executes rule graph (pure)
│  │  ├─ registry.py         # registers rules from packs
│  │  ├─ policy.py           # builds policy from config (currying)
│  │  ├─ dsl.py              # simple condition DSL (missing_pct > 0.9 etc.)
│  │  ├─ metrics.py
│  │  ├─ report.py
│  │  ├─ rescore.py
│  │  ├─ rules_builtin/
│  │  │  ├─ types.py         # coercions, datetime parse, category cast
│  │  │  ├─ missing.py       # numeric/cat/text imputation
│  │  │  ├─ datetime.py
│  │  │  ├─ outliers.py      # zscore/iqr detect + flag/winsorize/drop
│  │  │  ├─ text_norm.py     # trim, case, unicode normalize
│  │  │  ├─ units.py         # unit standardization (config-driven)
│  │  │  └─ prune.py         # drop sparse/constant columns, quarantine extras
│  │  └─ rules_custom/       # your future plug-ins (per project/dataset)
│  ├─ topics/
│  │  ├─ candidates.py
│  │  ├─ scoring.py
│  │  └─ select.py
│  ├─ layout/
│  │  ├─ planner.py
│  │  └─ chart_picker.py
│  ├─ chart/                 # <<< new: one home for all chart families
│  │  ├─ __init__.py
│  │  ├─ common.py           # theme, axes, ids, export (HTML/PNG)
│  │  ├─ change_over_time.py # Time charts
│  │  ├─ deviation.py        # diverging bar, spine, surplus/deficit
│  │  ├─ correlation.py      # scatter/bubble, xy-heatmap, col+line
│  │  ├─ ranking.py          # ordered bar, lollipop, slope, bump, strip
│  │  ├─ distribution.py     # hist/violin/box/
│  │  ├─ magnitude.py        # paired bars, radar, parcoords
│  │  ├─ part_to_whole.py    # stacked, treemap, pie (guarded)
│  │  ├─ spatial.py          # choropleth, proportional symbol
│  │  └─ flow.py             # sankey / (network extras later)
│  ├─ nlg/
│  │  ├─ narrative_constants.py
│  │  └─ narrative.py
│  ├─ nlg/
│  │  ├─ bootstrap.py
│  │  ├─ roles.py
│  │  ├─ schema_io.py
│  │  └─ schema.py
│  ├─ reports/
│  │  ├─ base.py
│  │  ├─ generators/
│  │  │  ├─ csv_report.py
│  │  │  ├─ json_report.py
│  │  │  ├─ parquet_report.py
│  │  │  ├─ chart_report.py
│  │  │  ├─ html_report.py
│  │  │  └─ pdf_report.py
│  │  └─ orchestrator.py
│  ├─ templating/
│  │  ├─ env.py
│  │  └─ templates/
│  │     ├─ base.html
│  │     ├─ section.html
│  │     ├─ components/
│  │     │  ├─ table.html
│  │     │  ├─ chart_embed.html
│  │     │  └─ badges.html
│  │     └─ styles/theme.css
│  ├─ gold/
│  │  └─ materialize.py
│  ├─ manifest/
│  │  └─ build.py
│  └─ utils/
│     ├─ fp.py
│     ├─ ids.py
│     ├─ time.py
│     └─ log.py
├─ scripts/
└─ tests/
```
