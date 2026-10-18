# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Canonicalise relative paths in recorded output hashes across host platforms."""
from pathlib import PurePosixPath


def canonical_output_hashes(values):
    result = {}
    for name, value in values.items():
        normal = name.replace('\\','/')
        path = PurePosixPath(normal)
        if path.is_absolute() or ':' in normal or '..' in path.parts or normal != path.as_posix():
            raise ValueError(f'Expected canonical relative output path: {name}')
        if normal in result:
            raise ValueError(f'Ambiguous output hash paths: {name}')
        result[normal] = value
    return result
