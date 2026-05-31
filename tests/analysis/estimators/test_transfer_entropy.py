"""Unit tests for transfer entropy estimator configs."""

import unittest

import pytest
from infomeasure.estimators.transfer_entropy import (
    DiscreteCTEEstimator,
    DiscreteTEEstimator,
    KernelCTEEstimator,
    KernelTEEstimator,
)

from nexus.analysis.constants import KernelType
from nexus.analysis.estimators.transfer_entropy import (
    DiscreteTransferEntropyConfig,
    KernelTransferEntropyConfig,
)
from tests.utils import Expected, Given, Scenario


class TestDiscreteTransferEntropyConfig(unittest.TestCase):
    """Unit tests for DiscreteTransferEntropyConfig."""

    @pytest.mark.unit
    def test_get_estimator_class(self) -> None:
        """Returns DiscreteTEEstimator as the estimator class."""
        self.assertIs(
            DiscreteTransferEntropyConfig().get_estimator_class(), DiscreteTEEstimator
        )

    @pytest.mark.unit
    def test_get_approach_name(self) -> None:
        """Returns 'discrete' as the approach name string used by infomeasure."""
        self.assertEqual(
            DiscreteTransferEntropyConfig().get_approach_name(), "discrete"
        )

    @pytest.mark.unit
    def test_get_conditional_estimator_class(self) -> None:
        """Returns DiscreteCTEEstimator as the conditional estimator class."""
        self.assertIs(
            DiscreteTransferEntropyConfig().get_conditional_estimator_class(),
            DiscreteCTEEstimator,
        )

    @pytest.mark.unit
    def test_get_estimator_kwargs(self) -> None:
        """Returns step_size, src/dest_hist_len, prop_time, and offset as kwargs."""
        scenarios = [
            Scenario(
                name="default",
                given=Given(data={"config": DiscreteTransferEntropyConfig()}),
                expected=Expected(
                    data={
                        "kwargs": {
                            "step_size": 1,
                            "src_hist_len": 1,
                            "dest_hist_len": 1,
                            "prop_time": 0,
                            "offset": 0,
                        }
                    }
                ),
            ),
            Scenario(
                name="custom",
                given=Given(
                    data={
                        "config": DiscreteTransferEntropyConfig(
                            step_size=2,
                            src_hist_len=3,
                            dest_hist_len=2,
                            prop_time=1,
                            offset=5,
                        )
                    }
                ),
                expected=Expected(
                    data={
                        "kwargs": {
                            "step_size": 2,
                            "src_hist_len": 3,
                            "dest_hist_len": 2,
                            "prop_time": 1,
                            "offset": 5,
                        }
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                self.assertEqual(
                    scenario.given.data["config"].get_estimator_kwargs(),
                    scenario.expected.data["kwargs"],
                )


class TestKernelTransferEntropyConfig(unittest.TestCase):
    """Unit tests for KernelTransferEntropyConfig."""

    @pytest.mark.unit
    def test_get_estimator_class(self) -> None:
        """Returns KernelTEEstimator as the estimator class."""
        self.assertIs(
            KernelTransferEntropyConfig().get_estimator_class(), KernelTEEstimator
        )

    @pytest.mark.unit
    def test_get_approach_name(self) -> None:
        """Returns 'kernel' as the approach name string used by infomeasure."""
        self.assertEqual(KernelTransferEntropyConfig().get_approach_name(), "kernel")

    @pytest.mark.unit
    def test_get_conditional_estimator_class(self) -> None:
        """Returns KernelCTEEstimator as the conditional estimator class."""
        self.assertIs(
            KernelTransferEntropyConfig().get_conditional_estimator_class(),
            KernelCTEEstimator,
        )

    @pytest.mark.unit
    def test_get_estimator_kwargs(self) -> None:
        """Returns all kernel TE config fields as estimator kwargs."""
        scenarios = [
            Scenario(
                name="default",
                given=Given(data={"config": KernelTransferEntropyConfig()}),
                expected=Expected(
                    data={
                        "kwargs": {
                            "step_size": 1,
                            "src_hist_len": 1,
                            "dest_hist_len": 1,
                            "prop_time": 0,
                            "offset": 0,
                            "bandwidth": 1.0,
                            "kernel": "gaussian",
                            "workers": 1,
                        }
                    }
                ),
            ),
            Scenario(
                name="custom",
                given=Given(
                    data={
                        "config": KernelTransferEntropyConfig(
                            step_size=2,
                            src_hist_len=3,
                            dest_hist_len=2,
                            prop_time=1,
                            offset=5,
                            bandwidth=0.5,
                            kernel=KernelType.BOX,
                            workers=2,
                        )
                    }
                ),
                expected=Expected(
                    data={
                        "kwargs": {
                            "step_size": 2,
                            "src_hist_len": 3,
                            "dest_hist_len": 2,
                            "prop_time": 1,
                            "offset": 5,
                            "bandwidth": 0.5,
                            "kernel": "box",
                            "workers": 2,
                        }
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                self.assertEqual(
                    scenario.given.data["config"].get_estimator_kwargs(),
                    scenario.expected.data["kwargs"],
                )

    @pytest.mark.unit
    def test_get_estimator_kwargs_kernel_is_string_not_enum(self) -> None:
        """kernel must be a plain string for infomeasure compatibility."""
        kwargs = KernelTransferEntropyConfig().get_estimator_kwargs()
        self.assertIsInstance(kwargs["kernel"], str)
