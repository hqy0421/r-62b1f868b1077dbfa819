import os
import shutil
import tempfile
import typing as tp

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pypeln as pl
from pypeln.thread.io import (
    DatasetNotFoundError,
    PartitionBoundaryError,
    garbage_collect,
)

SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("region", pa.string()),
        ("score", pa.float64()),
    ]
)


def make_rows(n: int, regions=("cn", "us")) -> tp.List[dict]:
    return [
        {
            "id": i,
            "region": regions[i % len(regions)],
            "score": None if i % 7 == 0 else float(i) / 3,
        }
        for i in range(n)
    ]


@pytest.fixture
def root():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def write(rows, root, **kwargs):
    defaults = dict(
        schema=SCHEMA,
        partition_columns=["region"],
        workers=3,
        rows_per_batch=37,
        rows_per_file=123,
    )
    defaults.update(kwargs)
    stage = pl.thread.to_parquet(rows, root=root, **defaults)
    pl.thread.run(stage)


def read(root, **kwargs):
    return list(pl.thread.from_parquet(root, **kwargs))


# ----------------------------------------------------------------
# basic roundtrip
# ----------------------------------------------------------------


def test_roundtrip_preserves_rows_types_nulls_and_duplicates(root):
    rows = make_rows(500)
    rows.append(dict(rows[0]))  # explicit duplicate row
    rows.append(dict(rows[0]))
    write(rows, root)

    out = read(root, workers=4, rows_per_batch=16)
    assert len(out) == len(rows)
    assert sorted(r["id"] for r in out) == sorted(r["id"] for r in rows)

    by_id = {r["id"]: r for r in out if r["id"] == 0}
    first = out[0]
    assert isinstance(first["id"], int)
    assert isinstance(first["score"], float) or first["score"] is None
    nulls = [r for r in out if r["score"] is None]
    assert nulls  # NULL preserved
    assert sum(1 for r in out if r["id"] == 0) == 3  # duplicates preserved


def test_unpartitioned_dataset(root):
    schema = pa.schema([("a", pa.int32()), ("b", pa.string())])
    rows = [{"a": i, "b": f"s{i}"} for i in range(50)]
    pl.thread.run(pl.thread.to_parquet(rows, root=root, schema=schema, workers=2))
    out = read(root)
    assert sorted(r["a"] for r in out) == list(range(50))


def test_missing_column_writes_null_unknown_column_raises(root):
    rows = [{"id": 1, "region": "cn"}, {"id": 2, "region": "us"}]
    write(rows, root)
    out = read(root)
    assert all(r["score"] is None for r in out)

    bad = pl.thread.to_parquet(
        [{"id": 1, "region": "cn", "unknown": 9}],
        root=root,
        schema=SCHEMA,
        partition_columns=["region"],
    )
    with pytest.raises(ValueError, match="not present in the declared schema"):
        pl.thread.run(bad)


def test_type_mismatch_raises_instead_of_implicit_cast(root):
    bad = pl.thread.to_parquet(
        [{"id": "not-an-int", "region": "cn", "score": 1.0}],
        root=root,
        schema=SCHEMA,
        partition_columns=["region"],
    )
    with pytest.raises((pa.ArrowTypeError, pa.ArrowInvalid)):
        pl.thread.run(bad)
    # nothing was published
    with pytest.raises(DatasetNotFoundError):
        read(root)


# ----------------------------------------------------------------
# append / overwrite semantics
# ----------------------------------------------------------------


def test_append_keeps_existing_data(root):
    write(make_rows(100), root, mode="append")
    write(make_rows(50), root, mode="append")
    out = read(root)
    assert len(out) == 150


def test_full_overwrite_replaces_everything(root):
    write(make_rows(100), root)
    write(make_rows(10), root, mode="overwrite", partitions=None)
    out = read(root)
    assert sorted(r["id"] for r in out) == list(range(10))


def test_partition_overwrite_only_touches_target_partition(root):
    write(make_rows(100), root)  # 50 cn, 50 us
    new_cn = [{"id": 1000 + i, "region": "cn", "score": 0.5} for i in range(7)]
    write(
        new_cn,
        root,
        mode="overwrite",
        partitions=[("region", "cn")],
    )
    out = read(root)
    assert len(out) == 57
    cn = [r for r in out if r["region"] == "cn"]
    us = [r for r in out if r["region"] == "us"]
    assert sorted(r["id"] for r in cn) == list(range(1000, 1007))
    assert len(us) == 50  # other partition untouched


def test_empty_input_overwrite_clears_target_partition(root):
    write(make_rows(100), root)  # 50 cn, 50 us
    write([], root, mode="overwrite", partitions=[("region", "cn")])
    out = read(root)
    assert len(out) == 50
    assert {r["region"] for r in out} == {"us"}


def test_empty_input_full_overwrite_clears_dataset(root):
    write(make_rows(40), root)
    write([], root, mode="overwrite")
    out = read(root)
    assert out == []


def test_overwrite_outside_target_partitions_raises(root):
    write(make_rows(100), root)
    before = read(root)
    bad_rows = [
        {"id": 999, "region": "us", "score": 1.0},
        {"id": 998, "region": "cn", "score": 1.0},
    ]
    bad = pl.thread.to_parquet(
        bad_rows,
        root=root,
        schema=SCHEMA,
        partition_columns=["region"],
        mode="overwrite",
        partitions={"region": "cn"},
        workers=1,
    )
    with pytest.raises(PartitionBoundaryError):
        pl.thread.run(bad)
    # old version fully preserved
    assert read(root) == before


def test_overwrite_multi_partition_boundary(root):
    write(make_rows(90), root)  # cn, us
    new_rows = [
        {"id": 500, "region": "cn", "score": 1.0},
        {"id": 501, "region": "eu", "score": 1.0},
    ]
    write(
        new_rows,
        root,
        mode="overwrite",
        partitions=[[("region", "cn")], [("region", "eu")]],
    )
    out = read(root)
    regions = {r["region"]: r["id"] for r in out if r["id"] >= 500}
    assert regions == {"cn": 500, "eu": 501}
    # us untouched
    assert len([r for r in out if r["region"] == "us"]) == 45


# ----------------------------------------------------------------
# transactional guarantees
# ----------------------------------------------------------------


def test_failed_write_preserves_previous_version_and_cleans_files(root):
    write(make_rows(100), root)
    manifest_before = open(os.path.join(root, "current.json")).read()

    def boom(_):
        raise RuntimeError("disk exploded")

    failing = (
        pl.thread.from_iterable(make_rows(20))
        | pl.thread.map(lambda r: boom(r))
        | pl.thread.to_parquet(
            root=root,
            schema=SCHEMA,
            partition_columns=["region"],
            workers=2,
        )
    )
    with pytest.raises(RuntimeError, match="disk exploded"):
        pl.thread.run(failing)

    assert open(os.path.join(root, "current.json")).read() == manifest_before
    assert sorted(read(root), key=lambda r: r["id"]) == sorted(
        make_rows(100), key=lambda r: r["id"]
    )
    # no unpublished files left behind
    txn_dir = os.path.join(root, "_transactions")
    if os.path.exists(txn_dir):
        leftovers = [
            os.path.join(dp, f) for dp, _, files in os.walk(txn_dir) for f in files
        ]
        assert leftovers == []


def test_explicit_abort_keeps_old_version(root):
    write(make_rows(60), root)
    before = sorted(read(root), key=lambda r: r["id"])

    import threading
    import time

    def slow_source():
        # slow enough that the abort timer fires mid-flight
        for row in make_rows(200):
            time.sleep(0.01)
            yield row

    stage = pl.thread.to_parquet(
        slow_source(),
        root=root,
        schema=SCHEMA,
        partition_columns=["region"],
        workers=1,
    )
    sink = stage.process_fn

    error = []

    def runner():
        try:
            for _ in stage:
                pass
        except BaseException as e:  # noqa: BLE001
            error.append(e)

    t = threading.Thread(target=runner)
    t.start()
    # the sink yields nothing, so abort from a separate thread while it runs
    time.sleep(0.2)
    sink.abort()
    t.join(timeout=30)
    assert not t.is_alive()
    assert error, "aborted transaction must surface an error to the pipeline"
    assert isinstance(error[0], RuntimeError)
    assert sorted(read(root), key=lambda r: r["id"]) == before


def test_reader_snapshot_isolation_during_overwrite(root):
    write(make_rows(200), root)
    stage = pl.thread.from_parquet(root, workers=2, rows_per_batch=8)

    iterator = iter(stage)
    first = [next(iterator) for _ in range(20)]

    # commit a brand new version while the reader is open
    write(make_rows(5), root, mode="overwrite")

    rest = list(iterator)
    seen = first + rest
    # old snapshot completed with its full 200 rows, not the new 5 rows
    assert len(seen) == 200
    assert sorted(r["id"] for r in seen) == list(range(200))

    # a fresh reader sees the new version
    assert len(read(root)) == 5


def test_partial_stage_style_and_run_true(root):
    sink = pl.thread.to_parquet(
        root=root,
        schema=SCHEMA,
        partition_columns=["region"],
        run=True,
    )
    result = sink(make_rows(30))
    assert result is None
    assert len(read(root)) == 30


# ----------------------------------------------------------------
# reader features
# ----------------------------------------------------------------


def test_partition_filter_and_columns(root):
    write(make_rows(100), root)
    cn = read(root, partitions=[("region", "cn")], columns=["id"])
    assert len(cn) == 50
    assert all(set(r.keys()) == {"id"} for r in cn)
    assert all("region" not in r for r in cn)


def test_reader_schema_validation(root):
    write(make_rows(10), root)
    other = pa.schema(
        [("id", pa.int64()), ("region", pa.int32()), ("score", pa.float64())]
    )
    with pytest.raises(ValueError, match="does not match"):
        read(root, schema=other)


def test_reading_never_written_dataset_raises(root):
    with pytest.raises(DatasetNotFoundError):
        read(root)


def test_multiple_runs_of_same_source_stage(root):
    write(make_rows(40), root)
    source = pl.thread.from_parquet(root)
    assert len(list(source)) == 40
    write(make_rows(10), root, mode="overwrite")
    # next run picks up the newest published version
    assert len(list(source)) == 10


# ----------------------------------------------------------------
# layout / configuration
# ----------------------------------------------------------------


def test_files_are_rolled_by_rows_per_file(root):
    write(make_rows(500), root, workers=4, rows_per_file=50)
    data_dir = os.path.join(root, "_data")
    parquet_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(data_dir)
        for f in files
        if f.endswith(".parquet")
    ]
    assert len(parquet_files) > 4
    for path in parquet_files:
        assert pq.read_metadata(path).num_rows <= 50


def test_hive_layout_on_disk(root):
    write(make_rows(20), root)
    assert os.path.isdir(os.path.join(root, "_data", "region=cn"))
    assert os.path.isdir(os.path.join(root, "_data", "region=us"))
    assert os.path.isdir(os.path.join(root, "_versions"))
    assert os.path.isfile(os.path.join(root, "current.json"))


def test_null_partition_value_roundtrip(root):
    schema = pa.schema([("id", pa.int64()), ("p", pa.string())])
    rows = [{"id": 1, "p": None}, {"id": 2, "p": "x"}]
    pl.thread.run(
        pl.thread.to_parquet(
            rows,
            root=root,
            schema=schema,
            partition_columns=["p"],
            workers=2,
        )
    )
    out = read(root)
    by_id = {r["id"]: r["p"] for r in out}
    assert by_id == {1: None, 2: "x"}
    nulls = read(root, partitions=[("p", None)])
    assert [r["id"] for r in nulls] == [1]


def test_garbage_collect_reclaims_old_files(root):
    schema = pa.schema([("id", pa.int64())])
    for i in range(3):
        pl.thread.run(
            pl.thread.to_parquet(
                [{"id": i}],
                root=root,
                schema=schema,
                mode="overwrite",
            )
        )
    current_files = []
    for dp, _, files in os.walk(os.path.join(root, "_data")):
        current_files.extend(files)
    assert len(current_files) == 3  # one per version so far

    deleted = garbage_collect(root, keep_versions=1)
    assert len(deleted) == 2
    # current version still fully readable
    assert read(root) == [{"id": 2}]
    remaining = []
    for dp, _, files in os.walk(os.path.join(root, "_data")):
        remaining.extend(files)
    assert len(remaining) == 1


def test_concurrency_and_buffer_settings_are_accepted(root):
    stage = pl.thread.to_parquet(
        make_rows(300),
        root=root,
        schema=SCHEMA,
        partition_columns=["region"],
        workers=8,
        maxsize=11,
        rows_per_batch=5,
        rows_per_file=33,
        compression="gzip",
    )
    pl.thread.run(stage)
    out = read(root, workers=5, maxsize=7, rows_per_batch=9)
    assert len(out) == 300


def test_invalid_arguments_raise(root):
    with pytest.raises(ValueError):
        pl.thread.run(pl.thread.to_parquet([], root=root, schema=SCHEMA, workers=0))
    with pytest.raises(ValueError):
        pl.thread.run(
            pl.thread.to_parquet([], root=root, schema=SCHEMA, mode="replace")
        )
    with pytest.raises(ValueError):
        pl.thread.run(
            pl.thread.to_parquet(
                [],
                root=root,
                schema=SCHEMA,
                partition_columns=["nope"],
            )
        )
    with pytest.raises(ValueError):
        pl.thread.from_parquet(root, partitions=[1, 2, 3])


# ----------------------------------------------------------------
# write lock
# ----------------------------------------------------------------


def test_write_lock_serializes_and_cleans_up(root):
    import threading
    import time

    from pypeln.thread.io.storage import WRITE_LOCK, DatasetStorage

    first = DatasetStorage(root)
    second = DatasetStorage(root)
    assert first.acquire_write_lock(timeout=5)

    order = []

    def contend():
        order.append("waiting")
        second.acquire_write_lock(timeout=10)
        order.append("acquired")
        second.release_write_lock()

    t = threading.Thread(target=contend)
    t.start()
    time.sleep(0.2)
    assert order == ["waiting"]

    first.release_write_lock()
    t.join(5)
    assert order == ["waiting", "acquired"]
    assert not os.path.exists(os.path.join(root, WRITE_LOCK))


def test_stale_lock_is_reclaimed_and_fresh_lock_blocks(root):
    import time

    from pypeln.thread.io.storage import WRITE_LOCK, DatasetStorage

    lock_path = os.path.join(root, WRITE_LOCK)
    with open(lock_path, "w") as f:
        f.write("stale-process")
    past = time.time() - 7200
    os.utime(lock_path, (past, past))

    storage = DatasetStorage(root)
    assert storage.acquire_write_lock(timeout=5)
    storage.release_write_lock()

    with open(lock_path, "w") as f:
        f.write("fresh-process")
    with pytest.raises(TimeoutError):
        DatasetStorage(root).acquire_write_lock(timeout=0.3)
    os.remove(lock_path)
