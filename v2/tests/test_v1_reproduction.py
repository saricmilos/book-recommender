"""The guard on everything else: v1's code, on the data in datasets/, must still give v1's
published numbers and the same split. Takes about half a minute."""

import pytest

from bookrec import evaluate

SPLIT_FINGERPRINT = "3ed51494707252f9"


@pytest.fixture(scope="module")
def results():
    return evaluate.run()


def test_v1_reproduces(results):
    failed = [row["what"] for row in results["reproduction"] if not row["match"]]
    assert failed == []


def test_split_is_unchanged(results):
    assert results["split"]["fingerprint"] == SPLIT_FINGERPRINT
