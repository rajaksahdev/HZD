"""The IMD server drops connections and serves empty files for unpublished days (Stage 8)."""

from datetime import date

import requests
from imdlib.util import DataNotAvailableError

from pipelines import run
from pipelines.sources import imd


def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(imd, "YEARLY", tmp_path / "yearly")
    monkeypatch.setattr(imd, "REALTIME", tmp_path / "realtime")
    monkeypatch.setattr(imd, "_open", lambda day: None)
    monkeypatch.setattr(imd.time, "sleep", lambda s: None)
    monkeypatch.setattr(imd.store, "log_download", lambda *a: None)


def test_unpublished_day_is_no_data_not_an_error(monkeypatch, tmp_path):
    offline(monkeypatch, tmp_path)

    def empty(*args, **kwargs):
        raise DataNotAvailableError("IMD returned an empty file")

    monkeypatch.setattr(imd.imdlib, "get_real_data", empty)
    monkeypatch.setattr(imd, "_is_yearly", lambda day: False)
    assert imd.fetch(date(2026, 11, 19)) == []
    assert run.process("imd", date(2026, 11, 19), fetch=True, force=False) == "no data"


def test_dropped_download_is_retried_and_its_partial_file_removed(monkeypatch, tmp_path):
    offline(monkeypatch, tmp_path)
    calls = []

    def flaky(*args, file_dir, **kwargs):
        calls.append(1)
        target = tmp_path / "yearly" / "rain" / "2005.grd"
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) < 3:
            target.write_bytes(b"half")  # what a reset connection leaves behind
            raise requests.ConnectionError("Connection aborted.")
        assert not target.exists()  # the half file was removed before this try
        target.write_bytes(b"whole")

    monkeypatch.setattr(imd.imdlib, "get_data", flaky)
    monkeypatch.setattr(imd, "_is_yearly", lambda day: True)
    new = imd.fetch(date(2005, 7, 26))
    assert len(calls) == 3
    assert [p.read_bytes() for p in new] == [b"whole"]
