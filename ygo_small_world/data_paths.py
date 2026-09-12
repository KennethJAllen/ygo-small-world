"""Locate source, bundled, and user-updated card databases independently of cwd."""

from pathlib import Path


def _source_root() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    if (root / 'pyproject.toml').is_file() and (root / 'ygo_small_world').is_dir():
        return root
    return None


def writable_card_path() -> Path:
    """Source checkouts update their snapshot; installations update a user copy."""
    root = _source_root()
    if root is not None:
        return root / 'data' / 'cardinfo.pkl'
    return Path.home() / '.ygo-small-world' / 'cardinfo.pkl'


def readable_card_path() -> Path:
    """Prefer the writable database, falling back to the packaged snapshot."""
    path = writable_card_path()
    if path.is_file() or _source_root() is not None:
        return path
    return Path(__file__).resolve().parent / 'data' / 'cardinfo.pkl'
