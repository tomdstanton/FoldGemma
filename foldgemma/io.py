"""Efficient IO module for reading and writing FASTA files in binary mode."""

import sys
from typing import BinaryIO, Iterable, Tuple


def read_fasta_bytes(file_handle: BinaryIO) -> Iterable[Tuple[bytes, bytes]]:
    """Yields (header_bytes, sequence_bytes) from a FASTA file efficiently.

    Args:
        file_handle: A binary file handle (e.g., opened with 'rb').

    Yields:
        Tuples of (header, sequence), both as bytes.
    """
    header = b""
    seq_chunks = []

    for line in file_handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(b">"):
            if header:
                yield header, b"".join(seq_chunks)
            header = line[1:]
            seq_chunks = []
        else:
            seq_chunks.append(line)
    
    if header:
        yield header, b"".join(seq_chunks)


def write_fasta_bytes(file_handle: BinaryIO, sequences: Iterable[Tuple[bytes, bytes]]) -> None:
    """Writes (header, sequence) byte pairs to a FASTA file.

    Args:
        file_handle: A binary file handle (e.g., opened with 'wb').
        sequences: An iterable of (header, sequence) byte tuples.
    """
    for header, seq in sequences:
        file_handle.write(b">" + header + b"\n" + seq + b"\n")


def get_binary_stdin() -> BinaryIO:
    """Returns the binary stream for standard input."""
    return sys.stdin.buffer


def get_binary_stdout() -> BinaryIO:
    """Returns the binary stream for standard output."""
    return sys.stdout.buffer
