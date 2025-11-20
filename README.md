Here’s a draft README you can drop straight into the repo (no code blocks, just text):

---

# Polymorphic Data Reporter

## Goal

Polymorphic Data Reporter is a Python project for building configurable, end-to-end reporting pipelines over tabular data.

The idea is to define your data sources, schemas, cleaning rules, and outputs once (via configs and schema definitions), and then re-use the same pipeline to:

* Ingest raw data into structured layers (e.g. raw → silver → gold tables)
* Clean and transform the data into an analysis-ready form
* Generate charts and visual summaries
* Produce natural-language style report text on top of those results

The “polymorphic” part is that the pipeline should be able to adapt to different datasets and reporting needs primarily through configuration and schemas rather than one-off, hard-coded scripts. The repository is structured to support running these pipelines locally, in containers, and eventually under orchestration (e.g. Airflow), with a focus on being well-tested and reproducible. ([GitHub][1])

## Tasks / Roadmap

Big-picture tasks for this project are:

1. Core project scaffolding

   * Maintain a clean src/tests/scripts layout with pyproject/poetry and requirements to support packaging and dependency management.
   * Keep utilities small, composable, and covered by tests.

2. Data I/O and synthetic data generation

   * Provide a reusable I/O layer to read and write tabular data from/to the filesystem (and eventually external systems).
   * Include a data generation script to create synthetic sample datasets into a raw data area for local experimentation.

3. Medallion-style data pipeline (raw → silver → gold)

   * Implement ingest logic that takes raw data and normalizes it into “silver” tables.
   * Build transformations that aggregate or reshape silver data into “gold” tables suitable for reporting.

4. Cleaning and feature engineering pipeline

   * Implement a dedicated cleaning pipeline for dataframes (null handling, type coercion, normalization, feature creation).
   * Keep the cleaning logic as pure, testable functions that can be reused across different report types.
   * Iterate on conditions and edge cases as they show up in synthetic or real data.

5. Analytics, charts, and reporting outputs

   * Generate charts/visualizations over the gold layer (e.g. time series, distributions, KPIs).
   * Expose a reporting step that pulls together tables, charts, and metrics into a cohesive “report” artifact.
   * Support multiple report “shapes” via configuration so new report types can be added without rewriting the pipeline.

6. NLP-based report narratives

   * Build an NLP layer that turns metrics and data slices into readable narrative summaries.
   * Wire those summaries into the final report output alongside charts and tables.

7. Orchestration and environments

   * Provide an Airflow-friendly layout and DAG(s) that run the main pipeline on a schedule.
   * Offer Docker support so the whole stack can run in a containerized environment, including dependencies and configs.

8. Documentation and examples

   * Maintain an up-to-date README that explains goals, tasks, and current status.
   * Add simple example configs and an example run that new users can copy as a starting point.

## Where the project is right now

As of the latest commits (through November 20, 2025), here’s the rough state of the world: ([GitHub][2])

1. Project structure and tooling

   * A multi-directory layout is in place (`src`, `tests`, `scripts`, `schemas`, `runs`, plus `airflow` and `docker` for orchestration/infra). ([GitHub][1])
   * Poetry and requirements management have been set up and iterated on, along with utility modules and tests for them. ([GitHub][2])

2. I/O layer and synthetic data

   * There is a data generation script that creates synthetic data into a `data/raw` area to drive the pipeline. ([GitHub][2])
   * I/O functionality and associated tests have been marked as “finished” in the commit history, so there’s a first pass of robust read/write behavior already implemented. ([GitHub][2])

3. Charts and visual reporting pieces

   * A first implementation of charting exists (“charts done!”), so the project can already produce visualizations from processed data. ([GitHub][2])
   * There is at least one commit specifically focused on adding/adjusting a “report” step, suggesting an initial reporting pipeline is wired up end-to-end. ([GitHub][2])

4. NLP and cleaning pipeline

   * NLP functionality has been started and brought close to completion (“nlp start”, then “nlp stuff almost done, going to cleaning”), which implies an early version of narrative generation over the data is present but still evolving. ([GitHub][2])
   * A cleaning pipeline for dataframes is actively being iterated on (“cleans pure function started on”, “clean finish hopefully + test”, “cleaning pipe being fixed”, and follow-up cleanup commits). Right now this area likely works for the core use case but is still being refined for edge cases and readability. ([GitHub][2])

5. Medallion-style ingest

   * There is an explicit commit that “ingest to silver works onto gold”, which suggests the raw → silver → gold flow is implemented at least for the primary target dataset(s). This is probably functional but may still need hardening and more configuration options as new use cases appear. ([GitHub][2])

6. Tests and quality

   * The commit history references tests for I/O, utilities, and cleaning, indicating that most of the core pieces are covered by at least a first pass of unit tests.
   * Multiple “cleanup”/“updates” commits show ongoing refactors and quality passes rather than only feature drops, which is a good sign for maintainability. ([GitHub][2])

7. Documentation

   * README content is actively being rewritten (“Delete existing README content”, followed by an “Update README.md” commit). This README text is meant to capture the high-level intent, roadmap, and current status so future documentation can build on a clear narrative. ([GitHub][2])

In short: the core pipeline skeleton, I/O, charting, NLP stubs, cleaning logic, and a medallion-like flow are all in place and under active iteration. The next big pushes are likely: tightening the cleaning and NLP behaviors, hardening configurations and Airflow/Docker integration, and adding more polished documentation and example runs so someone new can go from clone → config → generated report with minimal friction.

[1]: https://github.com/TylerEnglish/Polymorphic_Data_Reporter "GitHub - TylerEnglish/Polymorphic_Data_Reporter"
[2]: https://github.com/TylerEnglish/Polymorphic_Data_Reporter/commits/main/ "Commits · TylerEnglish/Polymorphic_Data_Reporter · GitHub"
