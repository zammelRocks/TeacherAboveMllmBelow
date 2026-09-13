from pathlib import Path

from kinematics_grading.domain import TrajectoryKey, is_gradable_step, parse_step_info


def test_trajectory_key_parses_realistic_path():
    p = Path("generated/exercice_2/round_3/from_correct_4/instance_025/03_main_trend_corrected.png")
    key = TrajectoryKey.from_path(p)
    assert key == TrajectoryKey(exercise=2, round=3, source_exercise=4, instance="instance_025")


def test_trajectory_key_none_for_incomplete_path():
    assert TrajectoryKey.from_path(Path("some/random/path.png")) is None


def test_is_gradable_step_excludes_endpoints_and_panel():
    assert not is_gradable_step("00_incorrect_submission.png")
    assert not is_gradable_step("06_reference_answer.png")
    assert not is_gradable_step("panel.png")


def test_is_gradable_step_includes_01_through_05():
    for n in range(1, 6):
        assert is_gradable_step(f"0{n}_whatever.png")


def test_parse_step_info_expected_progress_schedule():
    expected = {"01": 20, "02": 40, "03": 60, "04": 80, "05": 95}
    for step_id, progress in expected.items():
        info = parse_step_info(f"{step_id}_x.png")
        assert info.expected_progress == progress
