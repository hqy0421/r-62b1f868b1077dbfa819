"""
Versioned, transactional Parquet dataset storage built on top of
`pyarrow.fs.FileSystem`.

On-disk layout::

    <root>/
        current.json                 # pointer to the published version
        _versions/
            version-00000001.json    # immutable manifest of every version
            version-00000002.json
            ...
        _data/                       # data files of published versions
            <hive partition dirs>/<file>.parquet
        _transactions/
            <txn-id>/                # unpublished files of an in-flight write
                <hive partition dirs>/<file>.parquet

Publication protocol (only the pointer swap is observable to readers):

1. Worker files are written under ``_transactions/<txn-id>/`` where no reader
   ever looks.
2. Once every worker finished and closed its files, a new manifest containing
   the data-file relative paths is written under ``_versions/``.
3. ``current.json`` is replaced with the new pointer (temp file + ``move``).

If anything fails before the pointer is swapped the whole transaction
directory is deleted and the previous version stays untouched.
"""

import base64
import json
import os
import threading
import time
import typing as tp
import uuid

import pyarrow as pa
import pyarrow.fs as pafs

CURRENT_POINTER = "current.json"
VERSIONS_DIR = "_versions"
DATA_DIR = "_data"
TRANSACTIONS_DIR = "_transactions"
WRITE_LOCK = "_write.lock"
MANIFEST_PREFIX = "version-"
MANIFEST_SUFFIX = ".json"
MANIFEST_NUMBER_DIGITS = 10
DEFAULT_NULL_PARTITION_TOKEN = "__HIVE_DEFAULT_PARTITION__"
# lock files older than this are considered leftovers from a crashed process
LOCK_STALE_SECONDS = 60 * 60


class DatasetError(Exception):
    """Base class for dataset related errors."""


class PartitionBoundaryError(DatasetError):
    """Raised when an overwrite write contains rows outside of the declared
    target partitions."""


class TransactionInProgressError(DatasetError):
    """Raised when a second write is started on a coordinator while another
    write is still in progress."""


class DatasetNotFoundError(DatasetError):
    """Raised when opening a reader on a dataset that was never written."""


def _normalize_path(path: str) -> str:
    path = path.replace(os.sep, "/")
    # keep a leading slash (absolute local paths) while collapsing
    # duplicated separators and trimming the trailing one
    leading = "/" if path.startswith("/") else ""
    path = path.strip("/")
    while "//" in path:
        path = path.replace("//", "/")
    return leading + path


def resolve_filesystem(
    root: str, filesystem: tp.Optional[pafs.FileSystem] = None
) -> tp.Tuple[pafs.FileSystem, str]:
    """Resolve a ``(filesystem, normalized_root_path)`` pair.

    `root` may be a local path or any URI understood by
    :func:`pyarrow.fs.FileSystem.from_uri` (e.g. ``s3://bucket/prefix``).
    Relative local paths are resolved against the current working directory.
    """
    if filesystem is not None:
        return filesystem, _normalize_path(root)

    if "://" in root:
        filesystem, path = pafs.FileSystem.from_uri(root)
        return filesystem, _normalize_path(path)

    abs_path = os.path.abspath(root)
    return pafs.LocalFileSystem(), _normalize_path(abs_path)


def encode_schema(schema: pa.Schema) -> str:
    """Serialize a :class:`pyarrow.Schema` to a portable base64 string."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema):
        # writing zero batches still serializes the schema
        pass
    return base64.b64encode(bytes(sink.getvalue())).decode("ascii")


def decode_schema(blob: str) -> pa.Schema:
    reader = pa.ipc.open_stream(pa.BufferReader(base64.b64decode(blob)))
    return reader.schema


def join_path(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part]
    if not cleaned:
        return ""
    leading = "/" if parts[0].startswith("/") else ""
    joined = "/".join(cleaned)
    while "//" in joined:
        joined = joined.replace("//", "/")
    return leading + joined


def hive_partition_dir(
    partition_columns: tp.Sequence[str],
    values: tp.Tuple[tp.Any, ...],
    null_token: str = DEFAULT_NULL_PARTITION_TOKEN,
) -> str:
    """Build the Hive-style relative directory for a partition tuple."""
    segments = []
    for column, value in zip(partition_columns, values):
        if value is None:
            segments.append(f"{column}={null_token}")
        else:
            segments.append(f"{column}={value}")
    return "/".join(segments)


def partition_values_key(values: tp.Tuple[tp.Any, ...]) -> str:
    """Stable hashable/serializable representation of a partition tuple."""
    return json.dumps(list(values), separators=(",", ":"), default=str)


class Snapshot(tp.NamedTuple):
    """Immutable view of a published dataset version."""

    version: int
    schema: pa.Schema
    partition_columns: tp.Tuple[str, ...]
    files: tp.Tuple["ManifestFile", ...]


class ManifestFile(tp.NamedTuple):
    relative_path: str
    partition_values: tp.Optional[tp.Tuple[tp.Any, ...]]


class Manifest(tp.NamedTuple):
    version: int
    schema: pa.Schema
    partition_columns: tp.Tuple[str, ...]
    files: tp.Tuple[ManifestFile, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "schema": encode_schema(self.schema),
                "partition_columns": list(self.partition_columns),
                "files": [
                    {
                        "relative_path": f.relative_path,
                        "partition_values": list(f.partition_values)
                        if f.partition_values is not None
                        else None,
                    }
                    for f in self.files
                ],
            },
            default=str,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Manifest":
        payload = json.loads(blob)
        return cls(
            version=int(payload["version"]),
            schema=decode_schema(payload["schema"]),
            partition_columns=tuple(payload.get("partition_columns", [])),
            files=tuple(
                ManifestFile(
                    relative_path=item["relative_path"],
                    partition_values=tuple(item["partition_values"])
                    if item.get("partition_values") is not None
                    else None,
                )
                for item in payload["files"]
            ),
        )


class DatasetStorage:
    """Low level operations against the versioned dataset layout."""

    def __init__(
        self,
        root: str,
        filesystem: tp.Optional[pafs.FileSystem] = None,
        null_partition_token: str = DEFAULT_NULL_PARTITION_TOKEN,
    ):
        self.filesystem, self.root = resolve_filesystem(root, filesystem)
        self.null_partition_token = null_partition_token
        self.versions_path = join_path(self.root, VERSIONS_DIR)
        self.data_path = join_path(self.root, DATA_DIR)
        self.transactions_path = join_path(self.root, TRANSACTIONS_DIR)
        self.current_path = join_path(self.root, CURRENT_POINTER)
        self.lock_path = join_path(self.root, WRITE_LOCK)
        # in-process fallback / serialization for filesystems without
        # exclusive-create semantics
        self._process_lock = threading.Lock()

    # ------------------------------------------------------------------
    # fs helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self, path: str):
        info = self.filesystem.get_file_info(path)
        if info.type == pafs.FileType.NotFound:
            self.filesystem.create_dir(path, recursive=True)

    def _write_json_atomic(self, path: str, payload: str):
        tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
        with self.filesystem.open_output_stream(tmp_path) as stream:
            stream.write(payload.encode("utf-8"))
        self.filesystem.move(tmp_path, path)

    def _read_json(self, path: str) -> tp.Optional[dict]:
        info = self.filesystem.get_file_info(path)
        if info.type == pafs.FileType.NotFound:
            return None
        with self.filesystem.open_input_stream(path) as stream:
            return json.loads(stream.read().decode("utf-8"))

    def _delete_if_exists(self, path: str):
        info = self.filesystem.get_file_info(path)
        if info.type != pafs.FileType.NotFound:
            self.filesystem.delete_file(path)

    # ------------------------------------------------------------------
    # write lock
    # ------------------------------------------------------------------

    def acquire_write_lock(self, timeout: float = 30.0) -> bool:
        """Acquire the dataset exclusive write lock.

        Uses exclusive file creation so the lock works across threads and
        processes on a local filesystem. Filesystems without atomic
        exclusive-create (or any fs-level failure) fall back to an in-process
        lock, which still serializes writers inside one Python process.
        """
        self._process_lock.acquire()
        deadline = time.monotonic() + timeout
        payload = f"{os.getpid()}-{uuid.uuid4().hex}".encode("utf-8")
        while True:
            try:
                self._try_exclusive_lock(payload)
                return True
            except FileExistsError:
                if self._lock_is_stale():
                    self._delete_if_exists(self.lock_path)
                    continue
                if time.monotonic() >= deadline:
                    self._process_lock.release()
                    raise TimeoutError(
                        f"Timed out waiting for the dataset write lock at "
                        f"{self.lock_path}"
                    )
                time.sleep(0.05)
            except (OSError, pa.ArrowNotImplementedError, pa.ArrowIOError):
                # filesystem cannot provide an exclusive create: the process
                # lock already held serializes in-process writers
                return True

    def _try_exclusive_lock(self, payload: bytes):
        if isinstance(self.filesystem, pafs.LocalFileSystem):
            # O_EXCL gives true cross-process atomicity locally
            fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            return
        # generic fs: best effort exclusive create
        info = self.filesystem.get_file_info(self.lock_path)
        if info.type != pafs.FileType.NotFound:
            raise FileExistsError(self.lock_path)
        with self.filesystem.open_output_stream(self.lock_path) as stream:
            stream.write(payload)

    def _lock_is_stale(self) -> bool:
        info = self.filesystem.get_file_info(self.lock_path)
        if info.type == pafs.FileType.NotFound:
            return False
        age = time.time() - info.mtime.timestamp()
        return age > LOCK_STALE_SECONDS

    def release_write_lock(self):
        try:
            self._delete_if_exists(self.lock_path)
        finally:
            if self._process_lock.locked():
                self._process_lock.release()

    def exists(self) -> bool:
        return (
            self.filesystem.get_file_info(self.current_path).type
            != pafs.FileType.NotFound
        )

    # ------------------------------------------------------------------
    # manifests / pointer
    # ------------------------------------------------------------------

    def manifest_path(self, version: int) -> str:
        number = str(version).zfill(MANIFEST_NUMBER_DIGITS)
        return join_path(
            self.versions_path, f"{MANIFEST_PREFIX}{number}{MANIFEST_SUFFIX}"
        )

    def write_manifest(self, manifest: Manifest):
        self._ensure_dir(self.versions_path)
        self._write_json_atomic(
            self.manifest_path(manifest.version), manifest.to_json()
        )

    def read_manifest(self, version: int) -> Manifest:
        path = self.manifest_path(version)
        payload = self._read_json(path)
        if payload is None:
            raise DatasetNotFoundError(
                f"Manifest for version {version} not found at {path}"
            )
        return Manifest.from_json(json.dumps(payload))

    def read_current_version(self) -> tp.Optional[int]:
        payload = self._read_json(self.current_path)
        if payload is None:
            return None
        return int(payload["version"])

    def read_current_manifest(self) -> tp.Optional[Manifest]:
        version = self.read_current_version()
        if version is None:
            return None
        return self.read_manifest(version)

    def write_pointer(self, version: int):
        self._write_json_atomic(
            self.current_path,
            json.dumps({"version": version}, separators=(",", ":")),
        )

    def list_versions(self) -> tp.List[int]:
        info = self.filesystem.get_file_info(self.versions_path)
        if info.type == pafs.FileType.NotFound:
            return []
        selector = pafs.FileSelector(self.versions_path, recursive=False)
        versions = []
        for item in self.filesystem.get_file_info(selector):
            name = os.path.basename(item.path)
            if not (
                name.startswith(MANIFEST_PREFIX) and name.endswith(MANIFEST_SUFFIX)
            ):
                continue
            number = name[len(MANIFEST_PREFIX) : -len(MANIFEST_SUFFIX)]
            if number.isdigit():
                versions.append(int(number))
        return sorted(versions)

    # ------------------------------------------------------------------
    # transactions
    # ------------------------------------------------------------------

    def transaction_dir(self, txn_id: str) -> str:
        return join_path(self.transactions_path, txn_id)

    def create_transaction_dir(self, txn_id: str) -> str:
        txn_dir = self.transaction_dir(txn_id)
        self.filesystem.create_dir(txn_dir, recursive=True)
        return txn_dir

    def cleanup_stale_transactions(self):
        """Remove transaction directories left behind by crashed processes."""
        info = self.filesystem.get_file_info(self.transactions_path)
        if info.type == pafs.FileType.NotFound:
            return
        selector = pafs.FileSelector(self.transactions_path, recursive=False)
        for item in self.filesystem.get_file_info(selector):
            if item.type == pafs.FileType.Directory:
                self._delete_dir_best_effort(item.path)

    def transaction_file_path(
        self, txn_id: str, partition_dir: str, file_name: str
    ) -> str:
        txn_dir = self.transaction_dir(txn_id)
        if partition_dir:
            return join_path(txn_dir, partition_dir, file_name)
        return join_path(txn_dir, file_name)

    def data_file_path(self, partition_dir: str, file_name: str) -> str:
        if partition_dir:
            return join_path(self.data_path, partition_dir, file_name)
        return join_path(self.data_path, file_name)

    def publish_file(self, txn_relative_path: str) -> str:
        """Move a closed transaction file into ``_data`` and return its path
        relative to the data directory."""
        txn_full = join_path(self.transactions_path, txn_relative_path)
        relative = _normalize_path(
            txn_relative_path.split("/", 1)[1]
            if "/" in txn_relative_path
            else txn_relative_path
        )
        destination = join_path(self.data_path, relative)
        parent = destination.rsplit("/", 1)[0]
        self._ensure_dir(parent)
        self.filesystem.move(txn_full, destination)
        return relative

    def delete_transaction(self, txn_id: str):
        self._delete_dir_best_effort(self.transaction_dir(txn_id))

    def _delete_dir_best_effort(self, path: str):
        try:
            info = self.filesystem.get_file_info(path)
            if info.type != pafs.FileType.NotFound:
                self.filesystem.delete_dir(path)
        except Exception:
            # best effort cleanup must never mask the original error
            pass

    def absolute_data_path(self, relative_path: str) -> str:
        return join_path(self.data_path, relative_path)

    # ------------------------------------------------------------------
    # snapshot / gc
    # ------------------------------------------------------------------

    def open_snapshot(self) -> tp.Optional[Snapshot]:
        manifest = self.read_current_manifest()
        if manifest is None:
            return None
        return Snapshot(
            version=manifest.version,
            schema=manifest.schema,
            partition_columns=manifest.partition_columns,
            files=manifest.files,
        )

    def garbage_collect(self, keep_versions: int = 1) -> tp.List[str]:
        """Delete data files that no retained manifest references.

        The current version and up to `keep_versions` older versions are
        retained so readers which opened an older snapshot keep working.
        """
        versions = self.list_versions()
        if not versions:
            return []

        retained = set(versions[-max(keep_versions, 1) :])
        referenced: tp.Set[str] = set()
        for version in retained:
            manifest = self.read_manifest(version)
            referenced.update(f.relative_path for f in manifest.files)

        deleted: tp.List[str] = []
        info = self.filesystem.get_file_info(self.data_path)
        if info.type == pafs.FileType.NotFound:
            return deleted
        selector = pafs.FileSelector(self.data_path, recursive=True)
        for item in self.filesystem.get_file_info(selector):
            if item.type != pafs.FileType.File:
                continue
            relative = _normalize_path(item.path[len(self.data_path) :].lstrip("/"))
            if relative not in referenced:
                try:
                    self.filesystem.delete_file(item.path)
                    deleted.append(relative)
                except Exception:
                    pass

        # drop manifests of the reclaimed versions
        for version in versions:
            if version not in retained:
                self._delete_if_exists(self.manifest_path(version))

        return deleted
