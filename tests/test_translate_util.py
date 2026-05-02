from app.pipeline.translate._util import batches


def test_batches_evenly_divides():
    assert list(batches([1, 2, 3, 4, 5, 6], 2)) == [[1, 2], [3, 4], [5, 6]]


def test_batches_uneven_last_chunk_smaller():
    assert list(batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batches_empty_list():
    assert list(batches([], 5)) == []


def test_batches_single_chunk_when_n_larger():
    assert list(batches([1, 2], 10)) == [[1, 2]]
