"""
Tests for shared agent utilities: chunk_list.
"""

import pytest
from agent.agent_utils import chunk_list


def test_chunk_list_even_split():
    """Items divide evenly into chunks."""
    result = chunk_list([1, 2, 3, 4], 2)
    assert result == [[1, 2], [3, 4]]


def test_chunk_list_uneven_split():
    """Last chunk is smaller when items don't divide evenly."""
    result = chunk_list([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_list_size_larger_than_list():
    """Single chunk when size exceeds list length."""
    result = chunk_list([1, 2, 3], 10)
    assert result == [[1, 2, 3]]


def test_chunk_list_empty():
    """Empty input returns empty list — no LLM call will be made."""
    result = chunk_list([], 5)
    assert result == []


def test_chunk_list_size_one():
    """Each item becomes its own chunk."""
    result = chunk_list([1, 2, 3], 1)
    assert result == [[1], [2], [3]]