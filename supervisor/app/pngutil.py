"""Minimal PNG helpers (no Pillow): solid-colour images and header parsing."""

import struct
import zlib

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def solid_png(width: int, height: int, rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (
        PNG_SIG
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG header; raises ValueError if not a PNG."""
    if not data.startswith(PNG_SIG) or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])
