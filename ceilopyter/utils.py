import binascii
import datetime
import re
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from ceilopyter.common import InvalidMessageError


def read_hex(data: bytes, n_chars: int, n_gates: int) -> npt.NDArray[np.int32]:
    """Read backscatter values from hex-encoded two's complement values."""
    n_bits = n_chars * 4
    limit = (1 << (n_bits - 1)) - 1
    offset = 1 << n_bits
    x = np.frombuffer(data.upper(), dtype=np.uint8)
    is_digit = (x >= 48) & (x <= 57)  # 0-9
    is_letter = (x >= 65) & (x <= 70)  # A-F
    if not np.all(is_digit | is_letter):
        raise ValueError("Invalid hex")
    y = np.where(is_letter, x - 55, x - 48)
    out = np.zeros(n_gates, dtype=np.int32)
    for i in range(n_chars):
        out <<= 4
        out |= y[i::n_chars]
    out[out > limit] -= offset
    return out


def crc16(data: bytes) -> int:
    """Compute checksum similar to CRC-16-CCITT."""
    return binascii.crc_hqx(data, 0xFFFF) ^ 0xFFFF


def date_format_to_regex(pattern: bytes) -> re.Pattern:
    """Converts a date format string to a regex pattern."""
    mapping = {
        b"%Y": rb"\d{4}",
        b"%m": rb"0[1-9]|1[0-2]",
        b"%d": rb"0[1-9]|[12]\d|3[01]",
        b"%H": rb"[01]\d|2[0-3]",
        b"%M": rb"[0-5]\d",
        b"%S": rb"[0-5]\d",
        b"%f": rb"\d{6}",
    }
    for key, value in mapping.items():
        pattern = pattern.replace(key, b"(?P<" + key[1:] + b">" + value + b")")
    return re.compile(pattern)


def parse_file(
    content: bytes, pattern: re.Pattern
) -> Iterator[tuple[datetime.datetime, bytes]]:
    parts = re.split(pattern, content)
    for i in range(1, len(parts), pattern.groups + 1):
        timestamp = datetime.datetime(
            int(parts[i + pattern.groupindex["Y"] - 1]),
            int(parts[i + pattern.groupindex["m"] - 1]),
            int(parts[i + pattern.groupindex["d"] - 1]),
            int(parts[i + pattern.groupindex["H"] - 1]),
            int(parts[i + pattern.groupindex["M"] - 1]),
            int(parts[i + pattern.groupindex["S"] - 1]),
            int(parts[i + pattern.groupindex["f"] - 1])
            if "f" in pattern.groupindex
            else 0,
        )
        message = parts[i + pattern.groups]
        yield timestamp, message


def next_line(
    lines: Iterator[bytes],
    length: int | None = None,
    prefix: bytes | None = None,
    suffix: bytes | None = None,
) -> bytes:
    try:
        line = next(lines)
    except StopIteration:
        msg = "Expected another line"
        raise InvalidMessageError(msg) from None
    if prefix is not None:
        line = line.removeprefix(prefix)
    if suffix is not None:
        line = line.removesuffix(suffix)
    if length is not None and len(line) != length:
        msg = f"Expected {length} characters but got {len(line)} instead"
        raise InvalidMessageError(msg)
    return line
