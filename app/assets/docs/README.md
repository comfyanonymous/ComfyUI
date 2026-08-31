# Intended behaviour

This document defines how the local asset system should behave — the ruled behavioural contract the implementation is held to. It does not describe the implementation. Rules carry stable ids (S*, MC*, CQ*) so findings elsewhere can cite them.

This in-repo copy is canonical. The wider review package — as-built architecture, an adversarial confidence assessment, and black-box conformance evidence — lives in the [`ideation-sharing` review share](https://github.com/Comfy-Org/ideation-sharing/tree/synap5e/docs/asset-record-content-split/asset-record-content-split), which snapshots this contract.

## Data model

An asset record represents one user-visible entity. It owns its name, tags, `job_id`, `loader_path`, MIME type, extracted metadata, user metadata, and optional preview relationship.

A content row represents bytes at a storage location. It owns the path, byte size, modification time, hash, and missing state. Multiple asset records may reference one content row. Two content rows may have the same hash; hash uniqueness is never a database invariant.

The path is required and unique among content rows that are not missing. Missing rows retain their former path so the scanner can attempt recovery.

Names are labels and are not unique. Records are global; the local asset system has no per-user ownership boundary.

MIME type and extracted metadata belong to the asset record because the same bytes may be interpreted in different ways. Extracted metadata may grow when an extractor learns more, but an extractor must not silently remove previously recorded facts within the same extractor version.

`loader_path` is fixed when the record is created. It records how the registry classified the path at discovery time.

### Public API surface: no top-level `file_path` (ratified 2026-08-27)

Public asset responses — list, detail, create/upload, from-hash, and update — serve `display_name`
(the record's name) and `loader_path`, plus the record's dynamic `metadata`. They do not carry a
top-level `file_path` field. The value that field used to expose was a namespace-rooted storage
locator (for example `models/checkpoints/flux.safetensors`), never an absolute filesystem path, and
it is no longer public. A client that needs a path-shaped value uses `loader_path` and `display_name`
instead.

This is a boundary between public and internal, not a removal of the underlying fact. The extracted
`metadata` may still contain a `file_path` key when system metadata extraction records one; that key
lives inside `metadata` and is untouched by this ruling. The physical `AssetContent.path` on the
content row, other internal DTO fields, and preview path computation stay implementation detail and
are never served directly, regardless of this ruling.

Ruled 2026-08-27: `file_path` was served at the top level of every asset response through standalone
commit `0b28b50f`, which removed it from `schemas_out.Asset`, both response builders, and the
`Asset`/`AssetUpdated` OpenAPI schemas. The commit is deliberately isolated — `git revert 0b28b50f`
restores the field with nothing else to undo — so this ruling can be reverted alone if revisited.

## Missing content

Every asset whose content is absent remains visible and carries the client-visible `missing` tag. Missing state belongs to the content row, so all records that reference the same content become missing together.

Only a hash can recover a missing content row, with one narrow exception for rows that were never hashed. The scanner hashes the file now present at the missing path and recovers the row only when exactly one missing candidate for that path has the same hash. If no candidate matches, the scanner creates new content. If multiple candidates match, none recover.

A missing row whose hash is null — a file deleted before it was ever hashed, for example while every server was down during an off-to-on hashing transition — can never satisfy the hash comparison, so it gets its own narrower recovery rule (ratified 2026-08-31, commit `1bf21c86`): it recovers only when it is the single missing null-hash candidate at that path AND its recorded byte size and modification time exactly match the restored file's verified stat; the freshly computed hash is set on the row as part of recovery. Both conditions are load-bearing — the stat facts are the only recorded identity a never-hashed row has, so a same-path restore that does not preserve them creates new content instead, and the old row stays missing rather than risk recovering the wrong record.

Recovery does not require the old and new modification times to match. With hashing disabled, recovery is unavailable.

Missing rows and their asset records remain until explicitly deleted. Routine scans do not remove them.

## Hashing modes

The `--enable-asset-hashing` flag remains and defaults to off.

With hashing off, the scanner uses modification time and byte size to detect changes. A modification-time change with an unchanged byte size is treated as the same file: the stored file facts refresh and the stored hash is cleared, because without hashing the digest can no longer be vouched for. A change to both modification time and size means new content. A size change without a modification-time change is undefined behaviour.

With hashing on, the scanner uses modification time as the cheap change detector and a hash as the identity check. When the modification time changes, verification runs through the existing background seeding and enrichment work:

- If the hash is unchanged, refresh the stored file facts on the existing content row.
- If the hash changed, mark the old content missing and create a new content row and asset record for the path.

The system persists the previous hashing mode. On a transition from off to on, it hashes every live row without a hash and revalidates every live row that already has one. On a transition from on to off, it retains existing hashes as inert data.

Hashing must use a stable file snapshot. The worker checks file facts before and after hashing and accepts the digest only when they still match. Otherwise it retries later.

### Uploads while hashing is off (ratified 2026-08-26)

Upload dedup runs unconditionally in both hashing modes; only scanner/output hashing is gated by `--enable-asset-hashing`. Uploads always hash, use content-addressed filenames in hash-routed destinations, and deduplicate by bytes even while background hashing is off. Deduplication must know the digest before deciding whether to store the bytes, and the upload request is already reading the whole file.

From-hash lookup remains disabled while hashing is off.

## Filesystem changes

### 1. File removed outside the API

Mark the content missing and keep every asset record visible. Apply the hash recovery rule if a file later appears at the same path.

### 2. File edited in place

In hash mode, verify the new bytes. Refresh the existing content row if the hash is unchanged. If the hash differs, mark the old content missing and create new content and a new asset record for the path.

With hashing off, a change to both modification time and size takes the second branch. A modification-time change alone refreshes the existing row's file facts and clears its stored hash, per the hashing-modes rule above.

An asset record never changes from one byte identity to another. Existing history references therefore continue to point to the old, now-missing content instead of silently serving new bytes.

### 3. Deleted path reused

Path reuse has the same final state as an in-place edit: the old content is missing and the new bytes receive new content and a new asset record. The result must not depend on whether a scan observed the deletion before the new file appeared.

### 17. File moved or renamed outside the API

There is no filesystem move identity. Mark the old path missing and create new content and a new asset record at the new path.

### 20. Partial download under its final filename

The scanner skips known partial-download extensions. For other files, it records file facts during the walk, waits once per scan pass for a short stability floor, and checks the facts again before inserting.

Files that change between those checks are still being written and are not admitted. Such a file is parked on a bounded, de-duplicated watch list and reconsidered on the next scan. Each parked entry is retried a limited number of times before it is dropped; a dropped file is admitted normally whenever a later scan observes it in a stable state.

The watch list holds at most one entry per path and has a fixed maximum size. On overflow the oldest entry is evicted. Eviction is not data loss: an evicted path is rediscovered by ordinary directory traversal on any subsequent scan.

Retry is bounded by scan cadence, not by a timer. There is no background poller, and this rule makes no promise of settlement within any particular interval. In practice output roots are re-scanned after every prompt, so a partially-written output is normally admitted on the following generation. If a stalled partial file passes the check and later resumes, normal change detection splits it from the prematurely admitted content.

### 21. Symlinks and hardlinks

Keep the existing traversal rules: follow symlinks, store lexical paths, and retain the inode cycle guard. Different paths produce different content rows even when they resolve to the same inode or bytes.

### 22. Case and Unicode path forms

Store absolute, structurally normalised paths: relative segments and repeated separators collapse at the write boundary, so every stored path is the lexical absolute form. Never canonicalize case or Unicode; compare byte-for-byte. Filesystems that treat two case or Unicode spellings as the same path may therefore produce duplicate rows.

### 25. Registry changes

Classification is fixed at record creation. A newly visible path receives the registry classification in effect when discovered. A path that leaves all registered prefixes becomes missing. An existing in-scope path is not reclassified when the registry changes.

## Asset operations

### 4. Delete through the API

Delete the target asset record. Leave its content row and file intact. Do not soft-delete the record, retain a tombstone, or revive the deleted identity during later discovery.

Deleting a record never deletes any other record. A preview record the deleted asset nominated remains untouched; references to a preview clear only when the preview record itself is deleted. (Ratified 2026-08-26, superseding an earlier conditional-cascade rule: the preview reference points from the deleted record to its target, so deleting the pointer must not destroy the target.)

If the file remains discoverable, a later scan may create a fresh asset record. It must never recreate the deleted identity.

### 5. Rename

Renames always succeed. Duplicate names are allowed. Renaming does not change tags, classification, content identity, or storage path.

### 6. Upload the same bytes with the same name

Every upload mints a new asset record for its own request — the request's tags, user metadata, and preview nomination always land on that new record, whether or not identical bytes were already uploaded under the same name. This rule applies to every upload endpoint, including `/upload/image`.

Content, not records, is what gets deduplicated. When qualifying content already holds these bytes — live, stat-consistent, non-temp (see the [lookup rule](#27-lookup-and-dedup-against-missing-content)) — the new record points at that existing content row instead of the bytes being written again. When no content qualifies, the upload writes new content as normal. Either way, a new record is always created.

Uploads are therefore not idempotent: a client that retries an upload after a timeout accumulates a second record rather than being handed back the first. This is accepted deliberately — core has no idempotency protection anywhere, prompt submission and deletion included — and matches the shape already ruled for cached output (rule 10 below): a cache hit reuses content but still mints a new delivery record for the new caller. (Ratified 2026-08-28, superseding an earlier same-name reuse rule: the previous behaviour returned the existing record unchanged and silently discarded the second request's tags, metadata, and preview nomination — see [confidence.md](https://github.com/Comfy-Org/ideation-sharing/blob/synap5e/docs/asset-record-content-split/asset-record-content-split/confidence.md) finding 7 and commit `8e1bb7ad`.)

Uploads hash in both modes (see the ratified uploads rule above), so this content-level dedup applies regardless of the scanner hashing flag.

### `updated_at` semantics (ratified 2026-08-28)

An asset record's `updated_at` reflects only the last explicit user or API edit to that record: a rename, a user-metadata update, a MIME-type change, a preview nomination, or a manual tag add or remove.

It never advances for serving or downloading the asset (access time is a separate concern from edit time), for scanner enrichment filling in extracted metadata, for the automatic missing or recovered tag projection, for a content split or content retire, or for the preview-deleted foreign-key cascade that clears a `preview_id`. Minting a new record — an upload, a cached rerun, a content split — does not touch any other record's `updated_at`; the new record simply carries its own fresh timestamp from creation, and every existing record's last-explicit-edit time is untouched. (Superseding an earlier implementation detail where reading a record's content also updated `updated_at`; see [confidence.md](https://github.com/Comfy-Org/ideation-sharing/blob/synap5e/docs/asset-record-content-split/asset-record-content-split/confidence.md) finding 5 and commit `740fc6f0`.)

### 7. Upload the same bytes with a different name

When byte matching is available, create a new asset record with the requested name and point it to the existing content row. Do not write the bytes a second time. Without byte matching, create a new content row and write the upload normally.

### 8. Upload different bytes with the same name

Create a new asset record and new content. Both records keep the shared display name.

### 9. Byte-identical generated output

Every non-cached save event creates a new asset record and a new content row. Do not merge generated outputs automatically. Once hashed, equal hashes make the byte relationship visible without changing either identity.

### 10. Cached output

A fully cached rerun does not execute the save node or write a file. The asset layer creates a new asset record for the new prompt and points it to the existing content row for the cached file. It does not mutate the earlier asset record or content row.

If the cache is invalidated or unavailable, the save node executes normally. It writes a new counter-named file, and the ordinary generated-output rule creates new content and a new asset record.

#### S10.3 Runtime-expanded cached output

Whenever final history contains a runtime-expanded output locator, a child absent from the final executed-node set is treated exactly like any other cached output. It does not write a file. The asset layer creates a new delivery record that points to the existing content, and it does not mark earlier content or records missing or mutate them.

Execution must never be inferred from omission in a pre-execution cache announcement. A fully cached wrapper that produces no child locator creates no asset-registration event and is outside this rule's scope.

#### S10.4 Registration happens at the point of the write

The asset layer registers a produced output at the moment the producing node finishes, not after the
prompt completes. Classification is determined by control flow, never by comparing final history
against a set of executed nodes:

- A node that reaches output processing has, by definition, executed. Cache hits return earlier and
  never reach it. Such an output is `EXECUTED`.
- An output delivered through the cached-UI path is, by definition, `CACHED`.

For an `EXECUTED` output the asset layer creates a new content row for the path, marks any existing
live content row at that path missing, creates a new asset record carrying the current prompt's
`job_id`, and returns that record's identifier to the caller.

For a `CACHED` output the asset layer creates a new asset record pointing at the existing content row
with the new prompt's `job_id`, and mutates nothing else. Rule S10.3 continues to govern the case where
a fully cached wrapper produces no child locator: no locator means no registration event.

No rule in this document requires a hash in order to classify or register an output. Rule 9 mandates
new content for every non-cached save regardless of whether the bytes changed, so classification never
depends on comparing digests.

An identifier returned to a caller must refer to the record created for this write. Returning the
identifier of a record that describes earlier bytes at the same path is a defect, not a stale read.

Registration failure must not fail a generation. The failure is logged, the output entry carries no
asset identifier, and no partially-written asset row survives.

A cached delivery record's extracted metadata is a copy of the earliest existing record's metadata for
that content (`created_at` ascending, `id` ascending tiebreak) — derived facts about fixed bytes cannot
drift, so no re-extraction occurs on the normal path. When live content exists but no sibling record
does, the delivery record extracts metadata fresh from the file. (Ratified 2026-08-26.)

#### S10.5 Emission, and the single source of truth for output identity

Both output paths register at the moment the output is **emitted** — the moment it becomes visible to
anything outside the executor. Emission, not sending, is the boundary:

- The executed path emits when the producing node finishes processing its output.
- The cached path emits when a cached output is served.

At each emission the asset layer creates the appropriate record, and the resulting identifier is
attached to the emitted output. Registration and publication to history occur **unconditionally**;
only transmission to a connected client is conditional. An output must be registered whether or not a
client is attached, so no registration may sit behind a client-connection check.

**The database is the only source of truth for output identity.** An asset identifier must never be
stored in the execution cache. The cache holds output locators; identity is attached on the way out,
every time, on both paths. A cached replay therefore reports the identifier of the delivery record
created for the current prompt, never an identifier retained from the prompt that first produced the
file.

It follows that no component may reconstruct output classification after the fact by comparing final
history against a record of which nodes executed. Classification is known at emission and is recorded
there. Any parallel structure that exists solely to re-derive it afterwards is redundant and must be
removed rather than maintained alongside the authoritative path.

Rationale, recorded because it is the whole point of this rule: the pre-amendment design held the same
fact in four places — the database, an identifier baked into the cached UI, a separate set of executed
node identifiers, and a post-prompt reconstruction that diffed the latter two. Those representations
could and did disagree, which is precisely how an output came to report the identifier of a record
describing earlier bytes at the same path.

### S29. Settled properties and the single eventually-consistent field

Every asset property that a client can observe is settled before that asset becomes observable, with
exactly one exception: the content hash.

A client reading an asset through the assets API must never see a property that is absent merely
because work has not finished yet. Size, modification time, MIME type, extracted metadata, `job_id`,
tags, preview location and missing-state are all determined at the moment the record is created and are
correct from the record's first observable instant.

The content hash is the sole eventually-consistent field. It is the only property whose cost scales
with file size, and it is therefore the only one permitted to be absent on a record a client can
already see. A null hash means "not yet computed", never "this content has no hash".

Capabilities that require a hash — from-hash lookup, upload deduplication, and recovery of missing
content — are unavailable for an asset whose hash has not yet settled. This is the same reduced
capability already described for disabled hashing, applied for a bounded interval rather than
permanently. This covers freshly-registered outputs even in hashing-on mode, not only the
disabled-hashing case.

#### S29.1 Accepted exception: scanner-discovered assets

Assets discovered by the filesystem scanner are the one accepted violation of S29. A seeded record may
be observable before its metadata has settled. This is knowingly accepted for now and is recorded for
follow-up rather than treated as conforming behaviour.

Closing it requires distinguishing "not yet enriched" from "enrichment ran and found nothing", because
excluding un-enriched records without that distinction would hide, permanently, any asset whose
extraction fails. Outputs and uploads are not covered by this exception and must satisfy S29 in full.

#### S29.2 Output registration never hashes on the save path

Registering a produced output computes no content hash. The record is created with a null hash
unconditionally, and the background enrichment pass fills it afterwards when hashing is enabled. When
hashing is disabled the hash remains null, consistent with the disabled-hashing rules.

Hashing reads every byte of the file and its cost scales without bound with file size, while the save
path sits inside the execution loop. No output size may add hashing latency to a generation, and no
hashing failure may surface on the save path — a failed or unstable read is the enrichment pass's
problem, later.

Uploads are outside this rule and hash inline, always: deduplication must know the digest before
deciding whether to store the bytes, and the upload request is already reading the whole file.

#### S10.6 Two producers writing one path in a single prompt is unsupported

Two output producers that resolve to the same path within one prompt is unsupported. The resulting
database state is undefined and no rule in this document constrains it.

Registration occurs per producer at emission, so a prompt containing such a collision produces one
record per producer, and their relative order determines the final state. That order is not a contract.

This is not a licence to corrupt: each individual registration still obeys S10.4, and no operation may
fail or leave a partially-written row. Only the combined outcome is undefined.

Save nodes assign counter-based filenames precisely so that concurrent producers do not collide, so a
workflow reaching this state has bypassed the normal naming path.

### 11. Server restart

Persist asset and content rows across restarts. In-memory history remains transient. A record may retain a `job_id` whose prompt is no longer present in history.

Temp records are the exception and follow the startup cleanup rule below.

### 12. Temp and preview output

Startup temp cleanup deletes both temp files and the records and content rows that represent them. No missing temp entities remain afterward.

Cloud temp and preview lifetime is outside this document's scope.

### 28. Temp content shared by permanent records

Expiry belongs to the content location. Upload dedup and from-hash lookup exclude temp content, so permanent records can never point to content that startup temp cleanup will remove.

### 15. Same model bytes in two category locations

Create one asset record and one content row per location. Equal hashes may reveal that the bytes match. Do not merge the locations.

### 16. Byte-identical content from two local users

The local asset system has global records and no owner field. It does not isolate or duplicate records by user.

### 26. Legacy `/view` routes

Keep path-based `/view` forms unchanged. The asset-id form is canonical. The blake3 form follows the live-content lookup rule.

### 27. Lookup and dedup against missing content

Hash lookup, from-hash creation, and upload dedup consider only non-temp content whose file is present. A database row is not proof that bytes can be served.

When no qualifying content exists, uploads store the bytes they received and from-hash creation refuses. A missing sibling is never substituted for requested content.

### From-hash tie-breaking

When several qualifying content rows have the requested hash, choose the oldest by `created_at`, then by lexicographic `id`.

## Jobs and provenance

### 13. Querying a job's outputs

The asset API does not provide a job-to-outputs query. A record's `job_id` is informational. In-memory history remains the mechanism for showing a run's outputs during that session.

Cached reruns create a new asset record whose `job_id` is the new prompt. The record points to the existing content row because the save node did not execute and no new file was written. Earlier asset records keep their original `job_id` values.

### 14. Recording where a file came from

Each asset record's `job_id` identifies the prompt associated with that record's own creation event. The system does not infer provenance across records that share content.

Embedded PNG metadata is independent of the asset database. Save code writes it into the file, and the frontend may read it to restore a workflow.

## Concurrency and failure boundaries

### 18. File changes during hashing

Accept a digest only from a stable snapshot as defined in the hashing section.

### 19. Concurrent writers

Database constraints choose the winner when scanners, hooks, uploads, or multiple server processes race. Losing writers retry or discard their work. Do not add advisory locks.

### 23. Ambiguous recovery

If more than one missing content row matches a recreated file's hash, recover none of them.

### 24. Database replacement while running

Deleting or replacing the database file while the server is running is undefined behaviour.

## Operational limits (known, accepted)

### Write pressure and reader starvation

The asset database is SQLite with a single database-wide writer lock and no configured busy
timeout or lock-error handling on any route. Several paths hold or contend for that lock as a
matter of design: every upload writes its bytes and mints a delivery record; a same-path
overwrite retires and re-inserts content; execution outputs register per-emission during the
generation loop; a background enrichment pass fills hashes and metadata row by row; hash-serves
write access time to every record sharing the served content; and the upload dedup claim
deliberately holds the write lock across its filesystem re-check and metadata extraction — that
breadth is the correctness mechanism for the content-reuse race, not an accident.

Under sustained concurrent writes (demonstrated with a batch of eight parallel uploads), a
reader such as `GET /api/assets` can exceed SQLite's default five-second busy wait and surface
an unhandled `database is locked` error as HTTP 500. The failure is bounded and non-corrupting:
one request fails, a retry succeeds. The remedy — some combination of a configured busy timeout,
lock-error translation to 503, and ultimately WAL journal mode — is an operational decision that
applies to the whole database layer and is deliberately deferred to follow-up work rather than
patched piecemeal here.

## Migration

When startup finds the schema that predates the record/content split, delete that asset database and rebuild it with a full scan. Do not migrate rows into the new schema.

This intentionally discards data that a scan cannot rebuild, including manual tags, user metadata, preview assignments, API-created records, and `job_id` links. This is acceptable only while the asset API retains its do-not-rely-on-this-data warning.

## Code quality (non-runtime)

### CQ1. Production names and comments

Production names are domain-accurate. Comments exist only for non-obvious constraints. Production code contains no positional or temporal labels such as `_b`, `wave`, `cutover`, `pre-X`, or `_v2`; no changelog-style comments; no dead stubs; and no commented-out code.

Known violations are tracked separately and fixed by a separate de-slop effort. The work that ratified this rule introduces none and is not required to fix pre-existing violations.
