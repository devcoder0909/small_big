"""Generate extension icons using pure Python (zlib & struct)."""

import os
import struct
import zlib


def create_png(width: int, height: int, color: tuple[int, int, int, int]) -> bytes:
    """Create a minimal PNG file in bytes."""
    r, g, b, a = color
    line = bytes([0]) + bytes([r, g, b, a]) * width
    raw_data = line * height

    # PNG chunks
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    idat_compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b"IDAT" + idat_compressed)
    idat_chunk = struct.pack(">I", len(idat_compressed)) + b"IDAT" + idat_compressed + struct.pack(">I", idat_crc)

    iend_crc = zlib.crc32(b"IEND")
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    png_header = b"\x89PNG\r\n\x1a\n"
    return png_header + ihdr_chunk + idat_chunk + iend_chunk


def generate_all_icons():
    icons_dir = os.path.join("extension", "icons")
    os.makedirs(icons_dir, exist_ok=True)

    sizes = [16, 32, 48, 128]
    # Indigo color: #6366f1 (99, 102, 241, 255)
    color = (99, 102, 241, 255)

    for size in sizes:
        filename = os.path.join(icons_dir, f"icon{size}.png")
        png_data = create_png(size, size, color)
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Created {filename}")


if __name__ == "__main__":
    generate_all_icons()
