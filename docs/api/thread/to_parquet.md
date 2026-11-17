# pl.thread.to_parquet

::: pypeln.thread.to_parquet

## Transactional semantics

Every `to_parquet` run writes a new immutable dataset version:

1. Worker threads stream rows into private files under
   `_transactions/<txn-id>/`, buffering at most `rows_per_batch` rows each and
   rolling a new file every `rows_per_file` rows. The full input is never
   accumulated in memory.
2. A version is only published once every worker finished and closed all its
   Parquet files. Publication writes a new manifest under `_versions/` and
   atomically swaps `current.json`.
3. If the pipeline fails or is explicitly stopped before the pointer swap,
   the whole transaction directory is deleted and the previously published
   version stays readable. There is never a partially visible version.

### Append vs overwrite

* `mode="append"` keeps every existing file and adds the new files on top.
* `mode="overwrite"` without `partitions` replaces the whole dataset.
* `mode="overwrite"` with `partitions` replaces only the declared
  partitions; every other partition is untouched. An empty input still
  clears the declared partitions.
* Declaring `partitions` also enforces a boundary in both modes: a row whose
  partition values fall outside the declaration raises
  `PartitionBoundaryError` before anything is committed.

### Concurrency control

Writes to the same dataset are serialized with an exclusive write lock
(`_write.lock`, cross-process on local filesystems; an in-process lock on
filesystems without exclusive-create support). Stale locks left by crashed
processes are reclaimed after one hour.
