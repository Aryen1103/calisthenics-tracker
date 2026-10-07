import pytest

from reps import RepCounter, calculate_angle


@pytest.mark.parametrize("a, b, c, expected", [
    ([1, 0], [0, 0], [0, 1], 90),     # right angle
    ([1, 0], [0, 0], [-1, 0], 180),   # straight arm
    ([1, 0], [0, 0], [1, 1], 45),
    ([0, 1], [0, 0], [1, 0], 90),     # same angle with the points swapped
    ([1, 0], [0, 0], [1, -1], 45),    # reflex side is folded back below 180
])
def test_calculate_angle(a, b, c, expected):
    assert calculate_angle(a, b, c) == pytest.approx(expected)


def test_counts_a_full_bend_and_extension():
    reps = RepCounter()
    completed = [reps.update(angle) for angle in [170, 120, 80, 120, 170]]
    assert completed == [False, False, False, False, True]
    assert reps.count == 1 and reps.stage == "up"


def test_partial_reps_do_not_count():
    reps = RepCounter()
    for angle in [170, 100, 170, 95, 165]:  # never bends below 90
        reps.update(angle)
    assert reps.count == 0


def test_jitter_around_a_threshold_counts_once():
    reps = RepCounter()
    for angle in [170, 85, 92, 88, 91, 161, 158, 162, 159]:
        reps.update(angle)
    assert reps.count == 1


def test_starting_bent_counts_the_first_extension():
    # e.g. a pull-up from a dead hang filmed mid-movement
    reps = RepCounter()
    for angle in [80, 170, 80, 170]:
        reps.update(angle)
    assert reps.count == 2


def test_reset_and_custom_thresholds():
    reps = RepCounter(up_angle=150, down_angle=60)
    for angle in [155, 70, 155]:  # 70 is not below the custom 60
        reps.update(angle)
    assert reps.count == 0
    for angle in [50, 155]:
        reps.update(angle)
    assert reps.count == 1
    reps.reset()
    assert reps.count == 0 and reps.stage is None
