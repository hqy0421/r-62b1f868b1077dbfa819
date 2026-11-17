"""
`from_parquet` source stage for the thread pipeline.

A reader resolves the dataset pointer once per pipeline run: every worker of
the same run reads the same immutable snapshot. New commits swap the pointer
file, so a later run sees the new version while readers already iterating
keep their open Parquet file handles and finish the old snapshot.
"""

import threading
import typing as tp

import pyarrow as pa
import pyarrow.parquet as pq

from pypeln import utils as pypeln_utils

from ..stage import Stage
from ..worker import ProcessFn, Worker
from .coordinator import normalize_partitions
from .storage import (
    DatasetNotFoundError,
    DatasetStorage,
    ManifestFile,
    Snapshot,
    partition_values_key,
)


class ReadCoordinator:
    """Resolves and caches the published snapshot for a pipeline run."""

    def __init__(
        self,
        storage: DatasetStorage,
        filters: tp.Optional[tp.Sequence[tp.Any]],
        columns: tp.Optional[tp.Sequence[str]],
        rows_per_batch: int,
        schema: tp.Optional[pa.Schema],
    ):
        self.storage = storage
        self.filters = normalize_partitions(filters)
        self.columns = list(columns) if columns is not None else None
        self.rows_per_batch = rows_per_batch
        self.expected_schema = schema

        self._lock = threading.Lock()
        self._cached_version: tp.Optional[int] = None
        self._cached_snapshot: tp.Optional[Snapshot] = None

    def snapshot(self) -> Snapshot:
        with self._lock:
            version = self.storage.read_current_version()
            if version is None:
                raise DatasetNotFoundError(
                    f"No published version found for dataset at {self.storage.root}"
                )
            if self._cached_snapshot is None or self._cached_version != version:
                manifest = self.storage.read_manifest(version)
                snapshot = Snapshot(
                    version=manifest.version,
                    schema=manifest.schema,
                    partition_columns=manifest.partition_columns,
                    files=manifest.files,
                )
                self._cached_snapshot = snapshot
                self._cached_version = version
            snapshot = self._cached_snapshot

            if self.expected_schema is not None:
                if not self.expected_schema.equals(snapshot.schema):
                    raise ValueError(
                        "Declared reader schema does not match the dataset "
                        f"schema.\nDeclared: {self.expected_schema}\n"
                        f"Dataset: {snapshot.schema}"
                    )
            if self.columns is not None:
                unknown = set(self.columns).difference(snapshot.schema.names)
                if unknown:
                    raise ValueError(
                        f"columns {sorted(unknown)} are not part of the "
                        f"dataset schema {snapshot.schema.names}"
                    )
            return snapshot

    def selected_files(self, snapshot: Snapshot) -> tp.Tuple[ManifestFile, ...]:
        if self.filters is None:
            return snapshot.files
        allowed = {
            partition_values_key(tuple(value for _, value in partition))
            for partition in self.filters
        }
        return tuple(
            f
            for f in snapshot.files
            if f.partition_values is not None
            and partition_values_key(f.partition_values) in allowed
        )


class ParquetSource(ProcessFn):
    def __init__(self, coordinator: ReadCoordinator, workers: int):
        self.coordinator = coordinator
        self.workers = workers

    def __call__(self, worker: Worker, **kwargs):
        # resolve once per worker; the coordinator guarantees every worker of
        # the same run observes the identical version
        snapshot = self.coordinator.snapshot()
        files = self.coordinator.selected_files(snapshot)
        assigned = [
            f for index, f in enumerate(files) if index % self.workers == worker.index
        ]

        row_index = 0
        output_queues = worker.stage_params.output_queues
        filesystem = self.coordinator.storage.filesystem
        columns = self.coordinator.columns
        batch_size = self.coordinator.rows_per_batch

        for manifest_file in assigned:
            full_path = self.coordinator.storage.absolute_data_path(
                manifest_file.relative_path
            )
            parquet_file = pq.ParquetFile(full_path, filesystem=filesystem)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=columns
            ):
                for row in batch.to_pylist():
                    output_queues.put(
                        pypeln_utils.Element(index=(worker.index, row_index), value=row)
                    )
                    row_index += 1


def from_parquet(
    root: str,
    *,
    schema: tp.Optional[pa.Schema] = None,
    partitions: tp.Optional[tp.Sequence[tp.Any]] = None,
    columns: tp.Optional[tp.Sequence[str]] = None,
    workers: int = 1,
    maxsize: int = 0,
    rows_per_batch: int = 1000,
    filesystem: tp.Optional[pa.fs.FileSystem] = None,
) -> Stage[dict]:
    """
    Source stage that reads a versioned Parquet dataset written by
    :func:`to_parquet`.

    The dataset pointer is resolved when the pipeline starts: all workers of
    the same run read one immutable snapshot, even if another process commits
    a new version while the pipeline is running. Already open readers keep
    reading the old snapshot until they finish; the next run picks up the
    newest version.

    Duplicate rows, declared types and ``NULL`` values are preserved exactly
    as stored.

    ```python
    import pypeln as pl

    stage = pl.thread.from_parquet(
        "./lake/orders",
        partitions=[("region", "cn")],
        workers=2,
    )
    for row in stage:
        process(row)
    ```

    Arguments:
        root: Dataset root path or URI previously written by `to_parquet`.
        schema: Optional schema to validate against the published manifest.
            Raises ``ValueError`` on mismatch. When omitted the manifest
            schema is used.
        partitions: Optional partition filter. Only files whose partition
            values match one of the entries are read. Accepts the same forms
            as `to_parquet`'s `partitions` argument.
        columns: Optional subset of columns to read.
        workers: Number of concurrent reader worker threads. Files are
            distributed round-robin across workers.
        maxsize: Maximum number of rows buffered between pipeline stages.
        rows_per_batch: Number of rows decoded from Parquet per batch.
        filesystem: Optional explicit :class:`pyarrow.fs.FileSystem`.

    Returns:
        A source `Stage` yielding ``dict`` rows.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if rows_per_batch < 1:
        raise ValueError("rows_per_batch must be >= 1")
    if schema is not None and not isinstance(schema, pa.Schema):
        raise TypeError(f"schema must be a pyarrow.Schema, got {type(schema)}")

    storage = DatasetStorage(root=root, filesystem=filesystem)
    coordinator = ReadCoordinator(
        storage=storage,
        filters=partitions,
        columns=columns,
        rows_per_batch=rows_per_batch,
        schema=schema,
    )

    return Stage(
        process_fn=ParquetSource(coordinator=coordinator, workers=workers),
        workers=workers,
        maxsize=maxsize,
        timeout=0,
        total_sources=1,
        dependencies=[],
        on_start=None,
        on_done=None,
        f_args=[],
    )
