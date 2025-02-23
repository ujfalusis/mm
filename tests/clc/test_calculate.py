import logging
import pytest
import numpy as np
import typing
from mm import logconfig
from mm.clc.calculate import fibonacci, measure
from mm.structures.exceptions import NotEnoughDataRows

logging.config.dictConfig(logconfig.TEST_LOGGING)

@pytest.fixture
def fibonacciData() -> np.ndarray:
    # def gendata(rows: int) -> list[list[int]]:
    #     return np.array([list(map(lambda x: x * pow(10, i), [1, 2, 1, 2, 1])) for i in range(rows)])
        
    return [[ 3,  6,  2,  4, 10, 100],
            [ 4,  6,  2,  5, 20, 99],
            [10, 10,  6,  8, 30, 98],
            [ 8, 12,  8,  8, 40, 97],
            [ 8, 24,  2, 12, 50, 96],
            [12, 12, 12, 12, 60, 95],
            [12, 12,  6,  6, 70, 94]]

def test_fibonacci(fibonacciData: np.ndarray):

    fib = fibonacci(fibonacciData, 3)
    assert np.shape(fib) == (4, 3, 6)
    assert np.array_equal(fib[2, 2, :], np.array([12, 24, 2, 12, 50 + 60, 95]))

    fib = fibonacci(fibonacciData, 4)
    assert np.shape(fib) == (1, 4, 6)
    assert np.array_equal(fib[0, 3, :], np.array([12, 24, 2, 12, 50 + 60 + 70, 94]))


@pytest.fixture
def measureData() -> np.ndarray:
    return np.array(
        [[[ 3,  6,  2,  4, 10, 100]],
         [[ 4,  6,  2,  5, 20, 99]],
         [[10, 10,  6,  8, 30, 98]],
         [[ 8, 12,  8,  8, 40, 97]],
         [[ 8, 24,  2, 12, 50, 96]],
         [[12, 12, 12, 12, 60, 95]],
         [[12, 12,  6,  6, 70, 94]]])


def test_measure(measureData: np.ndarray):
    meas = measure(measureData, log = False, target = True, shift = 2)
    assert np.shape(meas[0]) == (5, 1, 5)
    assert np.shape(meas[1]) == (5, 3)
    assert np.array_equal(meas[0][0, 0, :], np.array([10 / 10, 6 / 10, 8 / 10, 30, 98]))
    assert np.array_equal(meas[1][0, :], np.array([10 / 10, 2 / 10, 1]))
    assert np.array_equal(meas[0][1, 0, :], np.array([12 / 8, 8 / 8, 8 / 8, 40, 97]))
    assert np.array_equal(meas[1][1, :], np.array([12 / 8, 2 / 8, 1]))

    meas = measure(measureData, log = True, target = True, shift = 2)
    assert np.shape(meas[0]) == (5, 1, 5)
    assert np.shape(meas[1]) == (5, 3)
    assert np.array_equal(meas[0][1, 0, :], np.concatenate([np.log(np.array([12 / 8, 8 / 8, 8 / 8])), [40], [97]]))
    # 12 / 8 -> log(3/2) -> cat: 6
    # 2 / 8 -> log(1/4) -> cat: 0
    assert np.array_equal(meas[1][1, :], np.array([6, 0, 1]))

