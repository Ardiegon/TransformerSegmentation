import pytest
import sys


def test_pytest():
    """
    Should always pass. 
    What is the answer to life the universe and everything?
    """
    answer = 42
    assert answer == 42

def test_configured():
    import pathlib
    import src.configs.paths as p
    assert p.ROOT_PATH == pathlib.Path(__file__).parents[2]