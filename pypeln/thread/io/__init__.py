"""
Transactional Parquet dataset IO for the Pypeln thread pipeline.

The subpackage is imported lazily from :mod:`pypeln.thread` so that
`pyarrow` is only required when the Parquet API is actually used.

Public API:

* :func:`to_parquet` — transactional, partitioned Parquet sink stage.
* :func:`from_parquet` — snapshot-consistent Parquet source stage.
* :func:`garbage_collect` — reclaim data files of reclaimed old versions.
* :class:`DatasetError`, :class:`PartitionBoundaryError`,
  :class:`DatasetNotFoundError`, :class:`TransactionInProgressError`.
"""

import typing as tp

from .coordinator import WriteCoordinator, normalize_partitions
from .reader import ReadCoordinator, from_parquet
from .storage import (
    DatasetError,
    DatasetNotFoundError,
    DatasetStorage,
    PartitionBoundaryError,
    TransactionInProgressError,
)
from .writer import to_parquet

__all__ = [
    "to_parquet",
    "from_parquet",
    "garbage_collect",
    "normalize_partitions",
    "WriteCoordinator",
    "ReadCoordinator",
    "DatasetStorage",
    "DatasetError",
    "PartitionBoundaryError",
    "DatasetNotFoundError",
    "TransactionInProgressError",
]


def garbage_collect(
    root: str,
    keep_versions: int = 1,
    filesystem=None,
) -> tp.List[str]:
    """Delete data files that no retained manifest references.

    The current version and up to `keep_versions` older versions are kept so
    readers which opened an older snapshot can finish. Returns the list of
    deleted relative data paths.
    """
    storage = DatasetStorage(root=root, filesystem=filesystem)
    return storage.garbage_collect(keep_versions=keep_versions)
