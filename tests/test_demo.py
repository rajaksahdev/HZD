"""Stage 8: the demo builds every input the later steps need, in the right order."""

from datetime import date

from scripts import demo


def test_rain_jobs_cover_every_label_year_and_both_forecast_seasons():
    jobs = demo.rain_jobs()
    imd_years = {int(args[1][:4]) for args, _ in jobs if args[-1] == "imd"}
    openmeteo_years = {int(args[1][:4]) for args, _ in jobs if args[-1] == "openmeteo"}
    assert set(demo.label_years()) <= imd_years
    assert set(demo.FORECAST_SEASONS) <= imd_years
    assert openmeteo_years == set(demo.FORECAST_SEASONS)
    for args, done in jobs:  # the done-file is the last day the job builds
        end = date.fromisoformat(args[3])
        assert done == demo.processed(args[-1], end)


def test_steps_run_inputs_before_models_before_runs():
    names = [name for name, _, _ in demo.STEPS]
    assert names == ["static", "rain", "tide", "forecast", "flood", "today", "replay"]


def test_demo_reports_never_overwrite_committed_docs():
    assert demo.OUT.parts[-2:] == ("data", "demo")
