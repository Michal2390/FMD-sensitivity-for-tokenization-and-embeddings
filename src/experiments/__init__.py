"""Experiments package.

Keep this package initializer lightweight. Import experiment classes directly
from their modules so unit tests do not require optional heavy dependencies
such as torch, matplotlib or pretty_midi at package import time.
"""

__all__: list[str] = []
