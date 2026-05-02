import time

from app import jobs


def test_job_dataclass_to_dict_roundtrip():
    j = jobs.Job(
        id="abc", item_id="1", item_name="Movie",
        target_lang="fr", provider="llm", mode="audio",
    )
    d = j.to_dict()
    assert d["id"] == "abc"
    assert d["status"] == "queued"
    assert d["error"] is None
    assert d["mode"] == "audio"


def test_list_jobs_empty():
    assert jobs.list_jobs() == []


def test_list_jobs_returns_newest_first():
    j1 = jobs.Job(id="1", item_id="a", item_name="A", target_lang="fr", provider="llm", mode="audio")
    j1.queued_at = 100.0
    j2 = jobs.Job(id="2", item_id="b", item_name="B", target_lang="fr", provider="llm", mode="audio")
    j2.queued_at = 200.0
    jobs._jobs[j1.id] = j1
    jobs._jobs[j2.id] = j2
    listed = jobs.list_jobs()
    assert [j.id for j in listed] == ["2", "1"]


def test_get_job_returns_none_for_unknown():
    assert jobs.get_job("nonexistent") is None


def test_max_jobs_eviction(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    # We need a "main loop" or submit() will raise; instead, simulate the dict
    # state directly to test eviction.
    for i in range(5):
        j = jobs.Job(id=f"j{i}", item_id="x", item_name="x",
                      target_lang="fr", provider="llm", mode="audio")
        jobs._jobs[j.id] = j
        while len(jobs._jobs) > jobs.MAX_JOBS:
            jobs._jobs.popitem(last=False)
    # Only the 3 most recent remain
    assert set(jobs._jobs.keys()) == {"j2", "j3", "j4"}


def test_submit_without_main_loop_raises():
    import pytest
    # _main_loop is None at module load; submitting should raise a clear error.
    monkeypatch_loop = jobs._main_loop
    jobs._main_loop = None
    try:
        with pytest.raises(RuntimeError, match="main loop"):
            jobs.submit(
                item_id="1", item_name="x", target_lang="fr",
                provider="llm", mode="audio",
                runner=lambda j: None,
            )
    finally:
        jobs._main_loop = monkeypatch_loop
