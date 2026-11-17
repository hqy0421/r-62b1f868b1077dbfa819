# pl.thread.from_parquet

::: pypeln.thread.from_parquet

## Snapshot consistency

The dataset pointer (`current.json`) is resolved once when the pipeline
starts. All worker threads of the same run therefore read one immutable
snapshot even if another process commits a new version while the pipeline is
running:

* a reader already iterating keeps its open Parquet file handles and
  finishes the old snapshot;
* a later run (or a newly created source stage) picks up the newest
  published version.

Duplicate rows, the declared column types and `NULL` values are preserved
exactly as stored.

## Partition filtering and column projection

* `partitions` restricts reading to the given partitions, e.g.
  `partitions=[("region", "cn")]` or `partitions={"region": "cn"}`.
* `columns` projects a subset of columns, e.g. `columns=["id"]`.
* Passing `schema` validates the declared schema against the manifest and
  raises `ValueError` on mismatch; without it the manifest schema is used.

## Garbage collection

Old versions keep their data files until explicitly reclaimed:

::: pypeln.thread.garbage_collect
