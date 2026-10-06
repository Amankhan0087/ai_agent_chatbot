import pytest

import storage


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()


def test_session_title_comes_from_first_user_message():
    sid = storage.create_session()
    storage.add_message(sid, "user", "How do I deploy for free?")
    storage.add_message(sid, "assistant", "Use Streamlit Community Cloud.")
    storage.add_message(sid, "user", "Thanks!")

    assert storage.list_sessions()[0]["title"] == "How do I deploy for free?"


def test_long_titles_are_truncated():
    sid = storage.create_session()
    storage.add_message(sid, "user", "x" * 200)

    title = storage.list_sessions()[0]["title"]
    assert len(title) == storage.TITLE_MAX_LEN
    assert title.endswith("…")


def test_messages_keep_order_and_roles():
    sid = storage.create_session()
    storage.add_message(sid, "user", "hi")
    storage.add_message(sid, "assistant", "hello")

    assert storage.get_messages(sid) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_most_recently_updated_session_is_listed_first():
    first, second = storage.create_session(), storage.create_session()
    storage.add_message(first, "user", "older")
    storage.add_message(second, "user", "newer")
    storage.add_message(first, "assistant", "bump first")

    assert [s["id"] for s in storage.list_sessions()] == [first, second]


def test_delete_session_removes_its_messages():
    sid = storage.create_session()
    storage.add_message(sid, "user", "bye")
    storage.delete_session(sid)

    assert storage.list_sessions() == []
    assert storage.get_messages(sid) == []


def test_memories_are_deduplicated_case_insensitively():
    added = storage.add_memories(["Aman is a Python developer", "  aman IS a python  developer ", ""])

    assert added == 1
    assert [m["fact"] for m in storage.list_memories()] == ["Aman is a Python developer"]


def test_delete_and_clear_memories():
    storage.add_memories(["fact one", "fact two", "fact three"])
    first = storage.list_memories()[0]
    storage.delete_memory(first["id"])
    assert len(storage.list_memories()) == 2

    storage.clear_memories()
    assert storage.list_memories() == []
