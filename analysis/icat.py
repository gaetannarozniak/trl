"""Display a PNG inline using the kitty graphics protocol.

kitty does not speak the iTerm2 inline-image protocol that `imgcat` emits, which is why that one prints a wall of
escape codes instead of a picture. This writes kitty's own protocol directly, so nothing has to be installed on
the host: the file is base64-encoded and sent in 4096-byte chunks, control keys on the first chunk only. The
image is scaled to the terminal width, which keeps a wide figure from overflowing.
"""

import base64
import os
import shutil
import sys
from pathlib import Path


CHUNK = 4096
# Fraction of the terminal width to fill. The figure is scaled to fit, so its pixel size has no bearing on how
# big it looks here; this is the knob. `ICAT_WIDTH=0.5 make icat` for smaller.
WIDTH = float(os.environ.get("ICAT_WIDTH", 0.6))


def icat(path):
    payload = base64.standard_b64encode(Path(path).read_bytes())
    chunks = [payload[i : i + CHUNK] for i in range(0, len(payload), CHUNK)]
    columns = max(20, int(shutil.get_terminal_size().columns * WIDTH))
    for n, chunk in enumerate(chunks):
        # a=T transmit and display, f=100 PNG, c= width in cells (rows follow from the aspect ratio)
        control = f"a=T,f=100,c={columns}," if n == 0 else ""
        sys.stdout.write(f"\033_G{control}m={int(n < len(chunks) - 1)};{chunk.decode()}\033\\")
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        icat(argument)
