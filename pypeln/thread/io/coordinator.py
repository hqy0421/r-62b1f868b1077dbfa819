"""
Write-side transaction coordination for the Parquet dataset sink.

A :class:`WriteCoordinator` owns one in-flight transaction at a time. Each
pipeline worker creates a :class:`TransactionWorker` through
:meth:`WriteCoordinator.begin`, buffers rows in bounded batches and rolls
Parquet files under an invisible ``_transactions/<txn-id>/`` directory.

Publication happens at most once, executed by the last worker that finishes
(after every ParquetWriter has been closed):

* all transaction files are moved into ``_data``,
* the new manifest is written,
* ``current.json`` is atomically swapped.

Any failure before the swap aborts the transaction: the transaction
directory is removed and the previously published version is preserved.
"""

import threading
import typing as tp
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from .storage import (
    DatasetStorage,
    Manifest,
    ManifestFile,
    PartitionBoundaryError,
    TransactionInProgressError,
    hive_partition_dir,
    join_path,
    partition_values_key,
)


def normalize_partitions(
    partitions: tp.Optional[tp.Sequence[tp.Any]],
) -> tp.Optional[tp.Tuple[tp.Tuple[tp.Tuple[str, tp.Any], ...], ...]]:
    """Normalize the user supplied target-partition declaration.

    Accepted forms::

        None                                  -> no boundary
        {"p": 1}                              -> single partition as dict
        [("p", 1)]                            -> single partition, pairs
        [("p", 1), ("q", 2)]                  -> single partition, pairs
        [{"p": 1}, {"p": 2}]                  -> multiple partitions
        [[("p", 1)], [("p", 2)]]              -> multiple partitions

    The normalized result is a tuple of partitions, each partition being a
    tuple of ``(column, value)`` pairs.
    """
    if partitions is None:
        return None

    if isinstance(partitions, dict):
        return (tuple(partitions.items()),)

    sequence = list(partitions)
    if not sequence:
        return ()

    def _is_pair(item) -> bool:
        return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)

    def _is_pairs(item) -> bool:
        return (
            isinstance(item, (list, tuple))
            and len(item) > 0
            and all(_is_pair(pair) for pair in item)
        )

    # single partition expressed as a flat sequence of pairs
    if all(_is_pair(item) for item in sequence):
        return (tuple(sequence),)

    # sequence of partitions (dicts or pair-sequences)
    if all(isinstance(item, dict) or _is_pairs(item) for item in sequence):
        normalized: tp.List[tp.Tuple[tp.Tuple[str, tp.Any], ...]] = []
        for item in sequence:
            if isinstance(item, dict):
                normalized.append(tuple(item.items()))
            else:
                normalized.append(tuple(item))
        return tuple(normalized)

    raise ValueError(
        "partitions must be a sequence of (column, value) pairs, e.g. "
        "[('region', 'cn')], a dict, or a sequence of such entries; got "
        f"{partitions!r}"
    )


class WriteCoordinator:
    """Process-wide state for a single :func:`to_parquet` sink stage."""

    def __init__(
        self,
        storage: DatasetStorage,
        schema: pa.Schema,
        partition_columns: tp.Sequence[str],
        mode: str,
        target_partitions: tp.Optional[tp.Sequence[tp.Sequence[tp.Tuple[str, tp.Any]]]],
        rows_per_batch: int,
        rows_per_file: int,
        compression: str,
    ):
        self.storage = storage
        self.schema = schema
        self.partition_columns = tuple(partition_columns)
        self.mode = mode
        self.target_partitions = normalize_partitions(target_partitions)
        self.rows_per_batch = rows_per_batch
        self.rows_per_file = rows_per_file
        self.compression = compression

        self._lock = threading.RLock()
        self._active = False
        self._holds_write_lock = False
        self._txn_id: tp.Optional[str] = None
        self._workers_total = 0
        self._workers_ended = 0
        self._failed = False
        self._poisoned = False
        self._failure: tp.Optional[BaseException] = None
        self._registered: tp.List[str] = []
        self._partition_rows: tp.Dict[str, int] = {}

        self._validate_schema()

    def _validate_schema(self):
        names = set(self.schema.names)
        for column in self.partition_columns:
            if column not in names:
                raise ValueError(
                    f"Partition column '{column}' is not part of the schema "
                    f"columns {self.schema.names}"
                )
        if self.mode not in ("append", "overwrite"):
            raise ValueError(f"mode must be 'append' or 'overwrite', got {self.mode!r}")
        if self.target_partitions is not None:
            expected = len(self.partition_columns)
            for partition in self.target_partitions:
                if len(partition) != expected:
                    raise ValueError(
                        f"Target partition {partition!r} must declare exactly "
                        f"{expected} partition columns {self.partition_columns}"
                    )
                declared = tuple(name for name, _ in partition)
                if declared != self.partition_columns:
                    raise ValueError(
                        f"Target partition columns {declared!r} do not match "
                        f"schema partition columns {self.partition_columns}"
                    )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def begin(self, worker_index: int, workers_total: int) -> "TransactionWorker":
        with self._lock:
            if self._poisoned:
                # a worker was force-stopped (timeout) or the transaction was
                # aborted: this transaction can never be committed, so every
                # (re)started worker fails immediately
                if self._failure is not None:
                    raise self._failure
                raise RuntimeError("Transaction already failed")
            if not self._active:
                # serialize the whole transaction against concurrent writers
                # of the same dataset (cross-process via the lock file)
                self.storage.acquire_write_lock()
                self._holds_write_lock = True
                self.storage.cleanup_stale_transactions()
                self._txn_id = uuid.uuid4().hex
                self._active = True
                self._workers_total = workers_total
                self._workers_ended = 0
                self._failed = False
                self._failure = None
                self._registered = []
                self._partition_rows = {}
                self.storage.create_transaction_dir(self._txn_id)
            elif self._txn_id is None:  # pragma: no cover - defensive
                raise TransactionInProgressError("Transaction is not initialized")
            return TransactionWorker(
                coordinator=self,
                worker_index=worker_index,
                txn_id=self._txn_id,
            )

    def register_file(self, relative_path: str):
        with self._lock:
            self._registered.append(relative_path)

    def record_partition(self, key: str, num_rows: int):
        with self._lock:
            self._partition_rows[key] = self._partition_rows.get(key, 0) + num_rows

    def poison(self, error: BaseException):
        """Permanently kill the transaction so it can never be committed."""
        with self._lock:
            if self._poisoned:
                return
            self._poisoned = True
            self._failed = True
            self._failure = error
            txn_id = self._txn_id
        if txn_id is not None:
            self.storage.delete_transaction(txn_id)
        self._release_write_lock()

    def worker_failed(self, error: BaseException):
        with self._lock:
            if self._failed:
                return
            self._failed = True
            self._failure = error
            txn_id = self._txn_id
        if txn_id is not None:
            self.storage.delete_transaction(txn_id)

    def raise_if_aborted(self):
        """Raise the recorded failure if the transaction is already dead."""
        with self._lock:
            if self._poisoned and self._failure is not None:
                raise self._failure

    def failure(self) -> tp.Optional[BaseException]:
        with self._lock:
            return self._failure

    def worker_ended(self) -> bool:
        """Called by every worker once its writers are closed.

        Returns ``True`` for the last worker, which is responsible for the
        final commit or cleanup.
        """
        with self._lock:
            self._workers_ended += 1
            return self._workers_ended >= self._workers_total

    def finalize(self):
        """Single-shot commit/abort executed by the last finishing worker."""
        with self._lock:
            if not self._active:
                return
            txn_id = self._txn_id
            assert txn_id is not None
            try:
                if self._failed:
                    self.storage.delete_transaction(txn_id)
                    if self._failure is not None:
                        raise self._failure
                    raise RuntimeError("Transaction failed")
                self._commit(txn_id)
            finally:
                self._active = False
                self._txn_id = None
                self._release_write_lock()

    def _release_write_lock(self):
        if self._holds_write_lock:
            self._holds_write_lock = False
            self.storage.release_write_lock()

    def abort(self):
        """Explicitly tear down a running transaction (e.g. pipeline stop).

        Poisons the transaction before deleting files so neither the current
        nor a restarted worker can publish the aborted version.
        """
        self.poison(RuntimeError("Parquet write transaction was explicitly aborted"))

    # ------------------------------------------------------------------
    # commit
    # ------------------------------------------------------------------

    def _commit(self, txn_id: str):
        previous = self.storage.read_current_manifest()
        previous_version = previous.version if previous is not None else 0

        # 1. move every closed transaction file into _data
        published: tp.List[ManifestFile] = []
        for txn_relative in list(self._registered):
            relative = self.storage.publish_file(txn_relative)
            partition_values = self._partition_values_from_relative(relative)
            published.append(
                ManifestFile(
                    relative_path=relative,
                    partition_values=partition_values,
                )
            )

        # 2. build the new manifest according to the write mode
        if self.mode == "overwrite":
            if self.target_partitions is None:
                kept: tp.List[ManifestFile] = []
            else:
                replaced = {
                    partition_values_key(partition)
                    for partition in self._target_partition_values()
                }
                kept = [
                    f
                    for f in (previous.files if previous else ())
                    if f.partition_values is None
                    or partition_values_key(f.partition_values) not in replaced
                ]
            files = tuple(kept) + tuple(published)
        else:
            files = (tuple(previous.files) if previous is not None else ()) + tuple(
                published
            )

        manifest = Manifest(
            version=previous_version + 1,
            schema=self.schema,
            partition_columns=self.partition_columns,
            files=files,
        )

        # 3. publish manifest then swap the pointer; readers only observe
        #    complete versions because the pointer move is the last step.
        self.storage.write_manifest(manifest)
        self.storage.write_pointer(manifest.version)

        # 4. remove the now empty transaction directory
        self.storage.delete_transaction(txn_id)

    def _target_partition_values(
        self,
    ) -> tp.Tuple[tp.Tuple[tp.Any, ...], ...]:
        assert self.target_partitions is not None
        return tuple(
            tuple(value for _, value in partition)
            for partition in self.target_partitions
        )

    def _partition_values_from_relative(
        self, relative_path: str
    ) -> tp.Optional[tp.Tuple[tp.Any, ...]]:
        if not self.partition_columns:
            return None
        directory = relative_path.rsplit("/", 1)[0] if "/" in relative_path else ""
        values: tp.List[tp.Any] = []
        segments = directory.split("/") if directory else []
        token = self.storage.null_partition_token
        for column, segment in zip(self.partition_columns, segments):
            prefix = f"{column}="
            if not segment.startswith(prefix):
                return None
            raw = segment[len(prefix) :]
            if raw == token:
                values.append(None)
            else:
                values.append(self._coerce_partition_value(column, raw))
        return tuple(values)

    def _coerce_partition_value(self, column: str, raw: str) -> tp.Any:
        field = self.schema.field(column)
        try:
            scalar = pa.scalar(raw, type=pa.string()).cast(field.type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return raw
        return scalar.as_py()

    def validate_partition_boundary(self, values: tp.Tuple[tp.Any, ...]):
        if self.target_partitions is None:
            return
        allowed = self._target_partition_values()
        if values not in allowed:
            raise PartitionBoundaryError(
                f"Row belongs to partition "
                f"{dict(zip(self.partition_columns, values))} which is outside "
                f"the explicitly declared target partitions "
                f"{[dict(zip(self.partition_columns, p)) for p in allowed]}"
            )


class TransactionWorker:
    """Bounded, file-rolling Parquet writer used by one pipeline worker."""

    def __init__(
        self,
        coordinator: WriteCoordinator,
        worker_index: int,
        txn_id: str,
    ):
        self.coordinator = coordinator
        self.worker_index = worker_index
        self.txn_id = txn_id

        self._writers: tp.Dict[str, pq.ParquetWriter] = {}
        self._file_counts: tp.Dict[str, int] = {}
        self._file_rows: tp.Dict[str, int] = {}
        # bounded buffer: rows are flushed as soon as rows_per_batch is hit,
        # the full input is never accumulated
        self._buffer: tp.List[dict] = []
        self._closed = False

    @property
    def storage(self) -> DatasetStorage:
        return self.coordinator.storage

    @property
    def schema(self) -> pa.Schema:
        return self.coordinator.schema

    @property
    def partition_columns(self) -> tp.Tuple[str, ...]:
        return self.coordinator.partition_columns

    def add_row(self, row: dict) -> bool:
        """Buffer one row; flush the buffer when the batch size is reached."""
        if self._closed:
            raise RuntimeError("Transaction worker is already closed")
        self._validate_row(row)
        self._buffer.append(row)
        if (
            self.coordinator.rows_per_batch > 0
            and len(self._buffer) >= self.coordinator.rows_per_batch
        ):
            self.flush()
            return True
        return False

    def add_rows(self, rows: tp.Iterable[dict]):
        for row in rows:
            self.add_row(row)

    def _validate_row(self, row: dict):
        if not isinstance(row, dict):
            raise TypeError(f"Parquet sink expects dict rows, got {type(row).__name__}")
        unknown = set(row).difference(self.schema.names)
        if unknown:
            raise ValueError(
                f"Row contains columns {sorted(unknown)} not present in the "
                f"declared schema {self.schema.names}"
            )

    def flush(self):
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._buffer.clear()
        self.write_table(table)

    def write_table(self, table: pa.Table):
        if self._closed:
            raise RuntimeError("Transaction worker is already closed")
        if table.num_rows == 0:
            return

        table = table.cast(self.schema) if table.schema != self.schema else table

        if not self.partition_columns:
            self._write_into_file("__unpartitioned__", table)
            return

        keys = [
            tuple(table.column(c)[i].as_py() for c in self.partition_columns)
            for i in range(table.num_rows)
        ]
        groups: tp.Dict[str, tp.List[int]] = {}
        for row_index, values in enumerate(keys):
            self.coordinator.validate_partition_boundary(values)
            key = partition_values_key(values)
            groups.setdefault(key, []).append(row_index)
            self.coordinator.record_partition(key, 1)

        for key, indices in groups.items():
            values = keys[indices[0]]
            partition_dir = hive_partition_dir(
                self.partition_columns,
                values,
                self.storage.null_partition_token,
            )
            self._write_into_file(partition_dir, table.take(indices))

    def _write_into_file(self, partition_dir: str, table: pa.Table):
        rows_per_file = self.coordinator.rows_per_file
        current_rows = self._file_rows.get(partition_dir, 0)

        if (
            partition_dir in self._writers
            and rows_per_file > 0
            and current_rows + table.num_rows > rows_per_file
        ):
            self._roll_file(partition_dir)

        if partition_dir not in self._writers:
            self._open_file(partition_dir)

        writer = self._writers[partition_dir]
        writer.write_table(table)
        self._file_rows[partition_dir] = (
            self._file_rows.get(partition_dir, 0) + table.num_rows
        )

        if rows_per_file > 0 and self._file_rows[partition_dir] >= rows_per_file:
            self._roll_file(partition_dir)

    def _open_file(self, partition_dir: str):
        file_index = self._file_counts.get(partition_dir, 0)
        file_name = (
            f"part-{self.worker_index:05d}-{file_index:05d}-"
            f"{uuid.uuid4().hex}.parquet"
        )
        path = self.storage.transaction_file_path(self.txn_id, partition_dir, file_name)
        parent = path.rsplit("/", 1)[0]
        self.storage._ensure_dir(parent)
        writer = pq.ParquetWriter(
            path,
            self.schema,
            compression=self.coordinator.compression,
            filesystem=self.storage.filesystem,
        )
        self._writers[partition_dir] = writer
        self._file_counts[partition_dir] = file_index + 1
        self._file_rows[partition_dir] = 0
        # register eagerly so a crash mid-write still owns the whole file
        relative = join_path(self.txn_id, partition_dir, file_name)
        self.coordinator.register_file(relative)

    def _roll_file(self, partition_dir: str):
        writer = self._writers.pop(partition_dir, None)
        if writer is not None:
            writer.close()
        self._file_rows[partition_dir] = 0

    def close(self):
        """Flush buffered rows and close every open Parquet file.

        Must run before publication; a file is only visible to the commit
        step once it has been closed here.
        """
        if self._closed:
            return
        self.flush()
        errors = []
        for writer in list(self._writers.values()):
            try:
                writer.close()
            except BaseException as e:  # pragma: no cover - defensive
                errors.append(e)
        self._writers.clear()
        self._closed = True
        if errors:
            raise errors[0]

    def end(self):
        """Close files and let the coordinator track worker completion."""
        self.close()
        return self.coordinator.worker_ended()

    def mark_failed(self, error: BaseException):
        """Best effort failure path: drop buffers, close handles, abort."""
        self._buffer.clear()
        for writer in list(self._writers.values()):
            try:
                writer.close()
            except BaseException:
                pass
        self._writers.clear()
        self._closed = True
        self.coordinator.worker_failed(error)
        try:
            if self.coordinator.worker_ended():
                self.coordinator.finalize()
        except BaseException:
            pass

    def poison(self):
        """Permanently kill the whole transaction (timeout/abort)."""
        self._buffer.clear()
        for writer in list(self._writers.values()):
            try:
                writer.close()
            except BaseException:
                pass
        self._writers.clear()
        self._closed = True
        self.coordinator.poison(
            RuntimeError("Parquet write worker was stopped before commit")
        )
