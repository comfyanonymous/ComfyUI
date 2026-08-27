# Development view

How the asset system's code is organised, the boundaries between its layers, and how it is tested.
See [logical.md](logical.md) for what the parts mean and [process.md](process.md) for what runs.

## Module inventory

| Module | Owns |
| --- | --- |
| `api/routes.py` | The `/api/assets` HTTP handlers, request validation glue, error envelopes |
| `api/schemas_in.py` | Pydantic models for request bodies and query params |
| `api/schemas_out.py` | Pydantic models for responses — the wire contract |
| `api/upload.py` | Multipart upload streaming to a temp staging file |
| `database/models.py` | SQLAlchemy ORM: `Asset`, `AssetContent`, `AssetTag`, `Tag`, `AssetMeta`, `AssetSystemState` |
| `database/queries/records.py` | Record/content CRUD, listing with filters/cursor/counts, shared tag-filter clauses |
| `database/queries/tags.py` | Tag usage listing and filtered tag counts |
| `services/ingest.py` | Upload, from-hash creation, in-place registration, executed/cached output registration |
| `services/asset_management.py` | Detail/update/delete flows, hash-to-path resolution for `/api/view` |
| `services/lookup.py` | The qualified-content iterator: which content rows are servable by hash |
| `services/tagging.py` | Tag add/remove with the protected-tag bucket |
| `services/hash_mode_state.py` | Persisted hashing mode, off-to-on transition queue and drain |
| `services/snapshot_hash.py` | Stable-snapshot blake3 hashing: digest plus the verified stat, as one observation |
| `services/metadata_extract.py` | Tiered metadata extraction (filesystem facts, safetensors headers, image dimensions) |
| `scanner.py` | Filesystem walk, seeding new content/records, enrichment pass, missing-state sync |
| `scanner_admission.py` | Two-stat stability gate and the bounded watch list for still-changing files |
| `scanner_changes.py` | Change detection, content splitting, missing-content recovery, verification drain |
| `seeder.py` | The background scan orchestrator: thread lifecycle, fast/enrich phases, pause/resume |
| `lifecycle.py` | Startup/shutdown: temp DB-row wipe, temp filesystem sweep, transition drain, seeder start |
| `mode.py` | The runtime hashing flag, initialised once from CLI args |
| `helpers.py` | Hash representation (`blake3:` prefix) and the SQL path-prefix predicate |

Integration points outside the package:

| File | Role |
| --- | --- |
| `execution.py` | Calls registration at output emission; keeps the execution cache id-free |
| `comfy_execution/asset_enrichment.py` | The pure adapters between execution outputs and the ingest registration API |
| `main.py` | Startup wiring: DB init under the file lock, `lifecycle.run_startup`, shutdown cleanup |

## Layering rules

- `api/routes.py` calls services; services call `database/queries`; queries touch only `database/models`.
  Routes do not open sessions against the models directly.
- The scanner family (`scanner`, `scanner_admission`, `scanner_changes`, `seeder`) sits beside the
  services and shares the query layer. Two imports in this family are deliberately nested inside
  functions to break import cycles (`scanner_admission` ↔ `scanner`, `lifecycle` → `seeder`); each
  carries a comment naming its cycle.
- `mode` is initialised exactly once at startup (`mode.init(args)` in `main.py`); reading it before
  initialisation raises rather than silently defaulting.
- `comfy_execution/asset_enrichment.py` is the only execution-side caller of the registration API, and
  it never mutates its inputs — both adapters deep-copy before enriching.

## The hashing-mode switch

`--enable-asset-hashing` (defined in `comfy/cli_args.py`, default off) reaches the system through
`mode.init(args)`; everything else reads `mode.hashing_enabled()`.

The flag gates scanner change-verification, enrichment hashing, and from-hash creation. It does not
gate uploads: uploads hash unconditionally in both modes, which is what makes upload dedup work with
background hashing off.

The *persisted* mode lives in the `AssetSystemState` table, separate from the runtime flag, so that
startup can detect an off-to-on transition and revalidate existing rows
(see [process.md](process.md#hashing-mode-transition)).

## Testing

| Suite | Pins |
| --- | --- |
| `tests-unit/assets_test/` | The service and query layer: upload/dedup, serving fail-closed, listing filters, tagging |
| `tests-unit/assets_test/test_intended_behaviour.py` | One test per behavioural scenario the system commits to |
| `tests-unit/assets_test/services/test_split_policy.py` | Change detection: when a file counts as changed, what a split carries |
| `tests-unit/assets_test/services/test_registration_primitives.py` | Executed/cached output registration contracts |
| `tests-unit/assets_test/services/test_admission_gate.py` | The two-stat stability gate and watch list |
| `tests-unit/app_test/` | Migration/ORM parity, multi-step downgrade, database file locking |
| `tests-unit/execution_test/test_enrich_output.py` | Emission-time registration and cache id-freedom |
| `tests-unit/execution_test/test_execute_reentry.py` | Cache purity across async node re-entry |

Run with `pytest tests-unit/assets_test/` (plus `tests-unit/app_test/` and
`tests-unit/execution_test/` for the full surface).

## Migrations

Alembic migrations live under `alembic_db/versions/`. The head revision creates the record/content
schema; its module docstring states exactly what a fresh install gets and what an upgrade discards.
A parity test compares the migrated schema against the ORM metadata in both directions, and a
multi-step downgrade test exercises the reverse chain.
