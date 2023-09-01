"""
Read IDX format.

Depends on numpy.

Functions:
    from_bytes(bytes) -> numpy.array
"""

import numpy as np


data_types = {
    0x08: np.ubyte,
    0x09: np.byte,
    0x0b: np.short,
    0x0c: np.int32,
    0x0d: np.float32,
    0x0e: np.double
}


def from_bytes(content: bytes) -> np.array:
    """
    Read IDX file

        Parameters:
            content (bytes): Data in IDX format

        Returns:
            data (numpy.array): The file data
    """

    data_type = data_types[content[2]]
    dimensions_count = content[3]

    shape = []
    for dimension in range(dimensions_count):
        offset = 4 + dimension * 4
        shape.append(int.from_bytes(content[offset:offset + 4]))

    data = content[4 + dimensions_count * 4:]

    return np.frombuffer(data, dtype=data_type).reshape(shape)
