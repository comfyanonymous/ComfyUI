# Asset scanner & storage system

The local asset system stores, identifies, and serves the files ComfyUI generates and receives —
scanner-discovered files on disk, uploaded images, and execution outputs. It records each user-visible
asset as a database row, tracks the bytes behind it as a separately-owned content row, and keeps a
background scanner reconciling the database against the filesystem. The `--enable-asset-hashing` flag
selects how the scanner decides whether a file's bytes have changed.

**State described:** the asset system as implemented in this repository. This document set is
versioned with the code it describes; every file describes the present state, in the present tense.
Code lives under `app/assets/`, with execution-time integration in `execution.py`,
`comfy_execution/asset_enrichment.py`, and `main.py`.

## Views

This documentation follows the 4+1 view model — five complementary projections of one system state.

| View | Answers |
|---|---|
| [Logical](logical.md) | The record/content data model, the entities, and the invariants that hold across them |
| [Process](process.md) | The scan/enrich pipeline, its concurrency model, and the content-row lifecycle state machine |
| [Development](development.md) | Where the code lives, the layer boundaries, and the two hashing modes as a code-level switch |
| [Physical](physical.md) | The SQLite database, the on-disk file layout, the file lock, and how bytes are served |
| [Scenarios](scenarios.md) | End-to-end journeys — upload, generate, cache-replay, edit, delete, recover — traced in both hashing modes |

## Core concepts, in one place

- **Record vs content.** An `Asset` is a user-visible record (name, tags, mime type, metadata,
  `job_id`, preview). An `AssetContent` is the bytes at a path (path, size, mtime, hash, missing
  state). Many records may point at one content row; a record's content outlives the record.
- **Path is the key, not hash.** At most one *live* content row exists per path
  (`uq_asset_contents_path_live`). Hashes are not unique — two paths holding identical bytes are two
  content rows with equal hashes.
- **Missing, not deleted.** When a file vanishes, its content row is marked `is_missing` rather than
  removed; every record pointing at it stays listed, carrying an automatic `missing` tag. Serving is
  fail-closed: a missing row is never served, even though it remains catalogued.
- **Two hashing modes.** With hashing off, the scanner detects change by mtime+size and cannot recover
  a returned file. With hashing on, it uses mtime as a cheap trigger and a blake3 hash as the identity
  check, enabling recovery of returned files and byte-level upload dedup. Uploads hash in both modes;
  the flag gates scanner and output hashing only.
- **Registration at emission.** Execution outputs are registered as they are emitted, not swept up
  after the prompt finishes. The execution cache holds an id-free copy of each node's UI output, so a
  cache hit mints a fresh delivery record instead of replaying a stale asset id.

## Scope

Covers the asset database, its scanner, the ingest/serving service layer, the `/api/assets` HTTP
surface, and the execution-time registration path. Out of scope: the model-folder registry beyond the
paths asset rows reference, frontend behaviour, and any remote/cloud asset storage.
