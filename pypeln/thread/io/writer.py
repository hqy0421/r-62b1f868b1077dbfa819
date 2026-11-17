"""
`to_parquet` sink stage for the thread pipeline.

Rows are streamed through native Pypeln worker threads. Each worker buffers
at most `rows_per_batch` rows in memory, rolls a new Parquet file every
`rows_per_file` rows and only participates in the commit after every file
handle it owns has been closed.
"""

import typing as tp
from queue import Empty

import pyarrow as pa

from pypeln import utils as pypeln_utils
from pypeln.utils import A

from ..stage import Stage
from ..worker import ProcessFn, Worker
from .coordinator import WriteCoordinator
from .storage import DatasetStorage


class ParquetSink(ProcessFn):
    def __init__(
        self,
        coordinator: WriteCoordinator,
        workers: int,
    ):
        self.coordinator = coordinator
        self.workers = workers

    def __call__(self, worker: Worker, **kwargs):
        txn_worker = self.coordinator.begin(
            worker_index=worker.index,
            workers_total=self.workers,
        )
        input_queue = worker.stage_params.input_queue
        try:
            while not input_queue.is_done():
                # a force-stopped sibling (timeout) poisons the transaction;
                # make sure a blocked/restarted worker observes it instead of
                # waiting on the queue forever while upstream keeps running
                self.coordinator.raise_if_aborted()
                try:
                    elem = input_queue.get(timeout=pypeln_utils.TIMEOUT * 100)
                except Empty:
                    # queue.Empty on timeout: loop back and re-check both the
                    # queue state and the poison flag
                    continue
                self.coordinator.raise_if_aborted()
                txn_worker.add_row(elem.value)

            self.coordinator.raise_if_aborted()
            is_last = txn_worker.end()
            if is_last:
                # all workers closed their files: publish the new version
                self.coordinator.finalize()
        except pypeln_utils.StopThreadException:
            # The supervisor timed the worker out and force-stopped it.
            # Poison the transaction so neither the restarted attempt below
            # nor any other worker can commit, stop the upstream input queue
            # (a stuck upstream producer thread would otherwise keep the
            # pipeline alive forever) and surface the failure through the
            # main queue. Returning normally lets the Worker machinery signal
            # completion exactly once instead of restarting this worker.
            txn_worker.poison()
            try:
                worker.stage_params.input_queue.kill()
            except Exception:
                pass
            failure = self.coordinator.failure()
            if failure is not None:
                worker.main_queue.raise_exception(failure)
            return
        except BaseException as e:
            txn_worker.mark_failed(e)
            raise

    def abort(self):
        self.coordinator.abort()


@tp.overload
def to_parquet(
    stage: tp.Union[Stage[A], tp.Iterable[A]],
    root: str,
    schema: pa.Schema,
    *,
    partition_columns: tp.Sequence[str] = (),
    mode: str = "append",
    partitions: tp.Optional[tp.Sequence[tp.Any]] = None,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    rows_per_batch: int = 1000,
    rows_per_file: int = 100_000,
    compression: str = "snappy",
    filesystem: tp.Optional[pa.fs.FileSystem] = None,
    run: bool = False,
) -> tp.Optional[Stage[None]]:
    ...


@tp.overload
def to_parquet(
    root: str,
    schema: pa.Schema,
    *,
    partition_columns: tp.Sequence[str] = (),
    mode: str = "append",
    partitions: tp.Optional[tp.Sequence[tp.Any]] = None,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    rows_per_batch: int = 1000,
    rows_per_file: int = 100_000,
    compression: str = "snappy",
    filesystem: tp.Optional[pa.fs.FileSystem] = None,
    run: bool = False,
) -> pypeln_utils.Partial[tp.Optional[Stage[None]]]:
    ...


def to_parquet(
    stage: tp.Union[
        Stage[A], tp.Iterable[A], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    root: tp.Optional[str] = None,
    schema: tp.Optional[pa.Schema] = None,
    *,
    partition_columns: tp.Sequence[str] = (),
    mode: str = "append",
    partitions: tp.Optional[tp.Sequence[tp.Any]] = None,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    rows_per_batch: int = 1000,
    rows_per_file: int = 100_000,
    compression: str = "snappy",
    filesystem: tp.Optional[pa.fs.FileSystem] = None,
    run: bool = False,
) -> tp.Union[tp.Optional[Stage[None]], pypeln_utils.Partial[tp.Optional[Stage[None]]]]:
    """
    Sink stage that writes the pipeline into a versioned, partitioned
    Parquet dataset using native Pypeln worker threads.

    The write is transactional: data files land in a private transaction
    directory and are only published once every worker finished and closed
    its files. Publication writes a new manifest and atomically swaps the
    dataset pointer, so concurrent readers either see the previous version
    or the new one, never a partial write. If the pipeline fails or is
    explicitly stopped before committing, the unpublished files are deleted
    and the previous version is preserved.

    ```python
    import pyarrow as pa
    import pypeln as pl

    schema = pa.schema([("id", pa.int64()), ("region", pa.string())])
    stage = pl.thread.to_parquet(
        rows,
        root="./lake/orders",
        schema=schema,
        partition_columns=["region"],
        mode="overwrite",
        partitions=[("region", "cn")],
        workers=4,
        rows_per_batch=500,
    )
    pl.thread.run(stage)
    ```

    Arguments:
        stage: A Stage or Iterable of ``dict`` rows.
        root: Dataset root path or URI (e.g. ``s3://bucket/dataset``).
        schema: Explicit declared :class:`pyarrow.Schema`. Rows missing a
            column write ``NULL``; unknown columns raise ``ValueError``.
            Types and NULLs are preserved as declared, no implicit casts.
        partition_columns: Columns used to lay out Hive-style partitions.
        mode: ``"append"`` adds a new version keeping existing files;
            ``"overwrite"`` replaces data. With `partitions` set only the
            declared partitions are replaced and the rest of the dataset is
            untouched. An empty input still clears the declared partitions.
        partitions: Explicit target partitions. Declaring them turns on
            boundary validation: any row whose partition values fall outside
            this set raises :class:`PartitionBoundaryError` before anything
            is committed (applies to both modes). In ``"overwrite"`` mode only
            these partitions are replaced and the rest of the dataset is
            untouched (an empty input still clears them); in ``"append"``
            mode it only constrains where rows may land. Accepts
            ``[("region", "cn")]``, ``{"region": "cn"}`` or a list of such
            entries.
        workers: Number of concurrent writer worker threads.
        maxsize: Maximum number of rows buffered between pipeline stages.
        timeout: Seconds before stopping a worker stuck on a single task.
            A stopped write is aborted: unpublished files are deleted and the
            previous version is preserved. ``0`` (default) means unbounded.
        rows_per_batch: Number of rows converted into one Arrow record
            batch per worker; bounds per-worker memory.
        rows_per_file: Target number of rows per Parquet file; a new file is
            rolled when the limit is reached. ``0`` means a single file per
            worker and partition.
        compression: Parquet compression codec passed to PyArrow.
        filesystem: Optional explicit :class:`pyarrow.fs.FileSystem`.
        run: When ``True`` the stage is executed immediately and ``None`` is
            returned.

    Returns:
        A sink `Stage` (or a `Partial` when `stage` is omitted), or ``None``
        when `run=True`.
    """
    if isinstance(stage, pypeln_utils.Undefined):
        if root is None or schema is None:
            raise TypeError("to_parquet requires 'root' and 'schema' arguments")
        return pypeln_utils.Partial(
            lambda s: to_parquet(
                s,
                root=root,
                schema=schema,
                partition_columns=partition_columns,
                mode=mode,
                partitions=partitions,
                workers=workers,
                maxsize=maxsize,
                timeout=timeout,
                rows_per_batch=rows_per_batch,
                rows_per_file=rows_per_file,
                compression=compression,
                filesystem=filesystem,
                run=run,
            )
        )

    if root is None or schema is None:
        raise TypeError("to_parquet requires 'root' and 'schema' arguments")
    if not isinstance(schema, pa.Schema):
        raise TypeError(f"schema must be a pyarrow.Schema, got {type(schema)}")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if rows_per_batch < 1:
        raise ValueError("rows_per_batch must be >= 1")
    if rows_per_file < 0:
        raise ValueError("rows_per_file must be >= 0")

    from ..api.to_stage import to_stage

    input_stage = to_stage(stage, maxsize=maxsize)

    storage = DatasetStorage(root=root, filesystem=filesystem)
    coordinator = WriteCoordinator(
        storage=storage,
        schema=schema,
        partition_columns=tuple(partition_columns),
        mode=mode,
        target_partitions=partitions,
        rows_per_batch=rows_per_batch,
        rows_per_file=rows_per_file,
        compression=compression,
    )
    process_fn = ParquetSink(coordinator=coordinator, workers=workers)

    sink = Stage(
        process_fn=process_fn,
        workers=workers,
        maxsize=maxsize,
        timeout=timeout,
        total_sources=input_stage.workers,
        dependencies=[input_stage],
        on_start=None,
        on_done=None,
        f_args=[],
    )

    if not run:
        return sink

    for _ in sink:
        pass
    return None
