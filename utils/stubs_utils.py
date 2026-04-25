""" 
A module for caching and retrieving computational results using stub files to disk.

This module provides utility functions to save and load intermediate processed results, which helps avoid redundant computations and speeds up development iterations.
"""

import os
import pickle


def save_stub(stub_path, object):
    """ 
    Save an object to a stub file.

    Args:
        stub_path (str): The file path to where the object should be saved.
        object: Any Python object that can be serialized with pickle.

    Returns:
        None
    """
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))

    if stub_path:
        with open(stub_path, 'wb') as f:
            pickle.dump(object, f)


def read_stub(read_from_stub, stub_path):
    """ 
    Read a previously saved object from a stub file from disk if available.

    Args:
        read_from_stub (bool): Whether to read from the stub file.

        stub_path (str): The file path to the stub file containing the object.

    Returns:
        object: The loaded Python object is successful, None otherwise.
    """

    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    return None
