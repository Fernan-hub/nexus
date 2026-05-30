"""Unit tests for mutual information estimator configs."""

import unittest

import pytest
from infomeasure.estimators.mutual_information import (
    DiscreteCMIEstimator,
    DiscreteMIEstimator,
    KernelCMIEstimator,
    KernelMIEstimator,
)

from nexus.analysis.constants import KernelType
from nexus.analysis.estimators.mutual_information import (
    DiscreteMutualInformationConfig,
    KernelMutualInformationConfig,
)
from tests.utils import Expected, Given, Scenario


class TestDiscreteMutualInformationConfig(unittest.TestCase):
    """Unit tests for DiscreteMutualInformationConfig."""

    @pytest.mark.unit
    def test_get_estimator_class(self) -> None:
        self.assertIs(
            DiscreteMutualInformationConfig().get_estimator_class(), DiscreteMIEstimator
        )

    @pytest.mark.unit
    def test_get_conditional_estimator_class(self) -> None:
        self.assertIs(
            DiscreteMutualInformationConfig().get_conditional_estimator_class(),
            DiscreteCMIEstimator,
        )

    @pytest.mark.unit
    def test_get_estimator_kwargs_excludes_normalize(self) -> None:
        """normalize is excluded because DiscreteMIEstimator hardcodes it internally."""
        kwargs = DiscreteMutualInformationConfig().get_estimator_kwargs()
        self.assertNotIn("normalize", kwargs)

    @pytest.mark.unit
    def test_get_estimator_kwargs(self) -> None:
        scenarios = [
            Scenario(
                name="default",
                given=Given(data={"config": DiscreteMutualInformationConfig()}),
                expected=Expected(data={"kwargs": {"offset": 0}}),
            ),
            Scenario(
                name="custom offset",
                given=Given(data={"config": DiscreteMutualInformationConfig(offset=3)}),
                expected=Expected(data={"kwargs": {"offset": 3}}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                self.assertEqual(
                    scenario.given.data["config"].get_estimator_kwargs(),
                    scenario.expected.data["kwargs"],
                )


class TestKernelMutualInformationConfig(unittest.TestCase):
    """Unit tests for KernelMutualInformationConfig."""

    @pytest.mark.unit
    def test_get_estimator_class(self) -> None:
        self.assertIs(
            KernelMutualInformationConfig().get_estimator_class(), KernelMIEstimator
        )

    @pytest.mark.unit
    def test_get_conditional_estimator_class(self) -> None:
        self.assertIs(
            KernelMutualInformationConfig().get_conditional_estimator_class(),
            KernelCMIEstimator,
        )

    @pytest.mark.unit
    def test_get_estimator_kwargs(self) -> None:
        scenarios = [
            Scenario(
                name="default",
                given=Given(data={"config": KernelMutualInformationConfig()}),
                expected=Expected(
                    data={
                        "kwargs": {
                            "offset": 0,
                            "normalize": False,
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
                        "config": KernelMutualInformationConfig(
                            offset=2,
                            normalize=True,
                            bandwidth=0.5,
                            kernel=KernelType.BOX,
                            workers=2,
                        )
                    }
                ),
                expected=Expected(
                    data={
                        "kwargs": {
                            "offset": 2,
                            "normalize": True,
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
        """kernel must be a plain string, not a KernelType enum, for infomeasure compatibility."""
        kwargs = KernelMutualInformationConfig().get_estimator_kwargs()
        self.assertIsInstance(kwargs["kernel"], str)
