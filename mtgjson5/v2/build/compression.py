"""
MTGJSON Compression Operations

Provides parallel compression of MTGJSON output files into multiple formats.
Optimized for high throughput using Intel ISA-L (if available), optimized presets,
and minimized thread contention.
"""

import bz2
import gzip
import io
import logging
import lzma
import os
import pathlib
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import TracebackType
from typing import Any, BinaryIO, Optional, Type, Union

# Attempt to import hardware-accelerated GZIP library
try:
    from isal import igzip
    HAS_ISAL = True
except ImportError:
    HAS_ISAL = False

from ...compiled_classes import MtgjsonStructuresObject
from ...v2.consts import (
    ALL_CSVS_DIRECTORY,
    ALL_DECKS_DIRECTORY,
    ALL_PARQUETS_DIRECTORY,
    ALL_SETS_DIRECTORY,
)

LOGGER = logging.getLogger(__name__)

COMPRESSION_CHUNK_SIZE = 16 * 1024 * 1024


class OptimizedStreamingCompressor:
    """
    Context manager that handles file opening for specific compression formats with optimized presets.
    """

    def __init__(
        self,
        output_path: pathlib.Path,
        fmt: str,
        original_filename: str,
    ):
        self.output_path = output_path
        self.fmt = fmt
        self.original_filename = original_filename
        self._file: Optional[Union[BinaryIO, gzip.GzipFile, bz2.BZ2File, lzma.LZMAFile, io.BytesIO, Any]] = None
        self._zip_buffer: Optional[io.BytesIO] = None

    def __enter__(self) -> "OptimizedStreamingCompressor":
        try:
            if self.fmt == "gz":
                if HAS_ISAL:
                    # Level 1 or 2 is extremely fast and provides decent ratio
                    # equivalent to standard gzip -6 but much faster
                    self._file = igzip.IGzipFile(
                        self.output_path, "wb", compresslevel=1
                    )
                else:
                    self._file = gzip.open(self.output_path, "wb", compresslevel=6)

            elif self.fmt == "bz2":
                # BZ2 is naturally slow/CPU heavy; compresslevel 9 is standard
                # Lowering this helps speed marginally but sacrificing ratio
                self._file = bz2.open(self.output_path, "wb", compresslevel=9)

            elif self.fmt == "xz":
                # Preset 3 offers great compression (very close to 6)
                # but significantly faster throughput. Standard lib defaults to 6.
                self._file = lzma.open(self.output_path, "wb", preset=3)

            elif self.fmt == "zip":
                # Zip is not naturally stream-writable for data bodies in this way without
                # slightly more complex logic, so buffering to RAM is the standard approach
                # in this pipeline.
                self._zip_buffer = io.BytesIO()
                self._file = self._zip_buffer

            else:
                raise ValueError(f"Unknown format: {self.fmt}")

        except Exception as e:
            LOGGER.error(f"Failed to initialize {self.fmt} for {self.original_filename}: {e}")
            # Raise so the caller knows this format failed immediately
            raise e

        return self

    def write(self, data: bytes) -> None:
        """Write chunk to the underlying compressor."""
        if self._file:
            self._file.write(data)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._file is not None:
            # Handle closing/finalizing
            if self.fmt == "zip" and self._zip_buffer:
                self._zip_buffer.seek(0)
                # compresslevel=1 gives decent speed for Zip which is usually just a container here
                with zipfile.ZipFile(
                    self.output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1
                ) as zf:
                    zf.writestr(self.original_filename, self._zip_buffer.read())
                self._zip_buffer.close()
                self._file = None
            else:
                self._file.close()
                self._file = None


def _compress_file_streaming_sequential(
    file: pathlib.Path,
    chunk_size: int = COMPRESSION_CHUNK_SIZE,
) -> list[tuple[bool, str]]:
    """
    Compress a file using streaming - reads input once, writes to all formats
    sequentially within the same thread.

    Args:
        file: File to compress
        chunk_size: Size of chunks to read/write

    Returns:
        List of (success, format) tuples
    """
    formats = ["gz", "bz2", "xz", "zip"]
    results: dict[str, bool] = {fmt: False for fmt in formats}
    compressors: list[OptimizedStreamingCompressor] = []

    # Initialize all valid compressors
    for fmt in formats:
        try:
            output_path = pathlib.Path(f"{file}.{fmt}")
            compressor = OptimizedStreamingCompressor(output_path, fmt, file.name)
            # Enter context manually so we can hold multiple open at once
            # pylint: disable=unnecessary-dunder-call
            compressor.__enter__()
            compressors.append(compressor)
            results[fmt] = True  # Tentatively true
        except Exception:
            results[fmt] = False

    try:
        # Read Chunk -> Write Chunk loop
        # We minimize I/O reads by reading a large chunk once into RAM
        # then pushing that bytes object to C-extensions immediately.
        with open(file, "rb") as f_in:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                for compressor in compressors:
                    try:
                        compressor.write(chunk)
                    except Exception:
                        # If a write fails mid-stream, we'll mark failure on exit
                        # but keep processing other formats
                        pass

    except Exception as e:
        LOGGER.error(f"Global reading error for {file.name}: {e}")
        # If input reading fails, everything fails
        for fmt in formats:
            results[fmt] = False

    # Clean up / Finalize files
    for compressor in compressors:
        try:
            compressor.__exit__(None, None, None)
        except Exception as e:
            LOGGER.error(f"Finalization failed for {file.name} [{compressor.fmt}]: {e}")
            results[compressor.fmt] = False

    return [(results[f], f) for f in formats]


def _get_compression_workers() -> int:
    """Get optimal number of compression workers based on CPU count."""
    cpu_count = os.cpu_count() or 4
    # Because we removed nested threads, we can afford to use more workers here.
    return max(2, min(12, cpu_count - 1))


def compress_files_parallel(
    files: list[pathlib.Path], max_workers: int | None = None
) -> dict[str, int]:
    """
    Compress multiple files in parallel using ThreadPoolExecutor.

    Args:
        files: List of files to compress
        max_workers: Maximum parallel workers (default: calculated based on CPU)

    Returns:
        Dict with compression statistics
    """
    workers = max_workers or _get_compression_workers()
    stats = {"total": len(files), "success": 0, "failed": 0}

    # Log capability
    if HAS_ISAL:
        LOGGER.debug("Intel ISA-L acceleration enabled for GZIP")
    else:
        LOGGER.debug("Standard GZIP (no ISA-L detected)")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit the optimized sequential compressor
        futures = {
            executor.submit(_compress_file_streaming_sequential, f): f for f in files
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                results = future.result()
                if all(success for success, _ in results):
                    stats["success"] += 1
                    LOGGER.info(f"Compressed {file.name}")
                else:
                    stats["failed"] += 1
                    failed_formats = [fmt for success, fmt in results if not success]
                    LOGGER.warning(f"Failed formats for {file.name}: {failed_formats}")
            except Exception as e:
                stats["failed"] += 1
                LOGGER.error(f"Compression job failed for {file.name}: {e}")

    return stats


def _compress_mtgjson_directory(
    files: list[pathlib.Path], directory: pathlib.Path, output_file: str
) -> None:
    """Legacy helper: Create a temporary directory of files to be compressed via Shell."""
    temp_dir = directory.joinpath(output_file)

    LOGGER.info(f"Creating temporary directory {output_file}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy(str(file), str(temp_dir))

    LOGGER.info(f"Compressing {output_file}")

    compression_commands: list[list[str | pathlib.Path]] = [
        ["tar", "-jcf", f"{temp_dir}.tar.bz2", "-C", temp_dir.parent, temp_dir.name],
        ["tar", "-Jcf", f"{temp_dir}.tar.xz", "-C", temp_dir.parent, temp_dir.name],
        ["tar", "-zcf", f"{temp_dir}.tar.gz", "-C", temp_dir.parent, temp_dir.name],
        ["zip", "-rj", f"{temp_dir}.zip", temp_dir],
    ]
    _compressor(compression_commands)

    LOGGER.info(f"Removing temporary directory {output_file}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def _compressor(compression_commands: list[list[str | pathlib.Path]]) -> None:
    """Execute a series of compression commands via subprocess."""
    for command in compression_commands:
        with subprocess.Popen(command, stdout=subprocess.DEVNULL) as proc:
            if proc.wait() != 0:
                LOGGER.error(f"Failed to compress {proc.args!s}")


def _compress_directory_python(
    files: list[pathlib.Path],
    output_base: pathlib.Path,
) -> list[tuple[bool, str]]:
    """
    Create archives of files in multiple formats using Python.
    """
    import tarfile

    results = []
    dir_name = output_base.name

    # tar.gz (Use isal via wrapping fileobj if possible, but standard tarfile module
    # allows 'fileobj' passing in read mode, harder in write mode. 
    # Sticking to standard for tar due to complexity, but lowering compress level).
    try:
        # compresslevel=6 is standard, lowered slightly to 4 for speed on bulk dirs
        with tarfile.open(f"{output_base}.tar.gz", "w:gz", compresslevel=4) as tar:
            for f in files:
                tar.add(f, arcname=f"{dir_name}/{f.name}")
        results.append((True, "tar.gz"))
    except Exception as e:
        LOGGER.error(f"tar.gz failed: {e}")
        results.append((False, "tar.gz"))

    # tar.bz2
    try:
        with tarfile.open(f"{output_base}.tar.bz2", "w:bz2", compresslevel=5) as tar:
            for f in files:
                tar.add(f, arcname=f"{dir_name}/{f.name}")
        results.append((True, "tar.bz2"))
    except Exception as e:
        LOGGER.error(f"tar.bz2 failed: {e}")
        results.append((False, "tar.bz2"))

    # tar.xz
    try:
        # Standard tarfile doesn't expose 'preset' directly in valid kwargs for open,
        # but does allow xz-specific format. However, for reliability, we use defaults here.
        with tarfile.open(f"{output_base}.tar.xz", "w:xz") as tar:
            for f in files:
                tar.add(f, arcname=f"{dir_name}/{f.name}")
        results.append((True, "tar.xz"))
    except Exception as e:
        LOGGER.error(f"tar.xz failed: {e}")
        results.append((False, "tar.xz"))

    # zip
    try:
        # compresslevel 1 for speed
        with zipfile.ZipFile(
            f"{output_base}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for f in files:
                zf.write(f, f"{dir_name}/{f.name}")
        results.append((True, "zip"))
    except Exception as e:
        LOGGER.error(f"zip failed: {e}")
        results.append((False, "zip"))

    return results


def compress_mtgjson_contents_parallel(
    directory: pathlib.Path, max_workers: int | None = None
) -> dict[str, int]:
    """
    Compress all files within the MTGJSON output directory using optimized parallel processing.

    Args:
        directory: Directory containing files to compress
        max_workers: Max parallel workers (default based on CPU count)

    Returns:
        Dict with compression statistics
    """
    workers = max_workers or _get_compression_workers()
    LOGGER.info(
        f"Starting optimized parallel compression on {directory.name} ({workers} workers)"
    )

    compiled_names = MtgjsonStructuresObject().get_all_compiled_file_names()

    # Identify Sets
    set_files = [
        f
        for f in directory.glob("*.json")
        if f.stem not in compiled_names and f.stem.isupper()
    ]
    
    # Identify Decks
    deck_files = list(directory.joinpath("decks").glob("*.json"))

    # Identify SQL files
    sql_dir = directory.joinpath("sql")
    if sql_dir.exists():
        sql_files = (
            list(sql_dir.glob("*.sql"))
            + list(sql_dir.glob("*.sqlite"))
            + list(sql_dir.glob("*.psql"))
        )
    else:
        sql_files = (
            list(directory.glob("*.sql"))
            + list(directory.glob("*.sqlite"))
            + list(directory.glob("*.psql"))
        )

    # CSV files
    csv_files = list(directory.joinpath("csv").glob("*.csv"))

    # Parquet files
    parquet_files = list(directory.joinpath("parquet").glob("*.parquet"))

    # Compiled files
    compiled_dir = directory.joinpath("Compiled")
    if compiled_dir.exists():
        compiled_files = list(compiled_dir.glob("*.json"))
    else:
        compiled_files = [
            f for f in directory.glob("*.json") if f.stem in compiled_names
        ]

    if compiled_files:
        LOGGER.info(f"Compressing {len(compiled_files)} large compiled files...")
        compress_files_parallel(compiled_files, max_workers=min(workers, 4))

    remaining_files = (
        set_files + deck_files + sql_files + csv_files + parquet_files
    )
    
    if remaining_files:
        LOGGER.info(f"Compressing {len(remaining_files)} standard files...")
        compress_files_parallel(remaining_files, max_workers=workers)

    archive_tasks = [
        (set_files, ALL_SETS_DIRECTORY),
        (deck_files, ALL_DECKS_DIRECTORY),
        (csv_files, ALL_CSVS_DIRECTORY),
        (parquet_files, ALL_PARQUETS_DIRECTORY)
    ]

    for files, output_name in archive_tasks:
        if files:
            LOGGER.info(f"Creating archive: {output_name}")
            _compress_directory_python(
                files,
                directory.joinpath(output_name),
            )

    LOGGER.info("Finished parallel compression.")
    return {"total": len(remaining_files) + len(compiled_files), "success": 0, "failed": 0}
