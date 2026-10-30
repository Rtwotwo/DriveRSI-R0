"""Run in the isolated DriveRL environment with python -m unittest."""

import unittest

try:
    from driversi_agents.driverl.closed_loop import (
        native_tensor_equal,
        pack_objects,
        unpack_objects,
    )
except ImportError:
    unpack_objects = pack_objects = None


@unittest.skipIf(unpack_objects is None, "optional official DriveRL runtime")
class NativeObservationTests(unittest.TestCase):
    def test_source_angular_motion_and_unknown_are_distinct(self):
        for kind in ["VEHICLE", "PEDESTRIAN", "BICYCLE"]:
            for rate in [None, 0.0, -0.31, 0.75]:
                row = dict(
                    token="observation",
                    track_token="identity",
                    kind=kind,
                    pose=[12.0, -3.0, 0.23],
                    size=[4.0, 2.0, 1.5],
                    velocity=[3.0, 1.0],
                    angular_velocity=rate,
                )
                self.assertEqual(pack_objects(unpack_objects([row], 100000)), [row])

    def test_native_nan_is_preserved_without_becoming_zero(self):
        import json
        import math

        import torch

        row = dict(
            token="t",
            track_token="a",
            kind="PEDESTRIAN",
            pose=[0.0, 0.0, 0.0],
            size=[0.6, 0.6, 1.7],
            velocity=[0.0, 0.0],
            angular_velocity="nan",
        )
        result = unpack_objects([row], 100000)
        self.assertTrue(math.isnan(result.tracked_objects.tracked_objects[0].angular_velocity))
        self.assertEqual(json.loads(json.dumps(pack_objects(result), allow_nan=False)), [row])
        self.assertTrue(
            native_tensor_equal(
                torch.tensor([float("nan"), 1.0]), torch.tensor([float("nan"), 1.0])
            )
        )
        self.assertFalse(
            native_tensor_equal(torch.tensor([float("nan"), 1.0]), torch.tensor([0.0, 1.0]))
        )
        self.assertFalse(
            native_tensor_equal(
                torch.tensor([float("nan"), 1.0]), torch.tensor([float("nan"), 1.0001])
            )
        )

    def test_old_records_without_angular_velocity_remain_unknown(self):
        row = dict(
            token="t",
            track_token="a",
            kind="VEHICLE",
            pose=[0.0, 0.0, 0.0],
            size=[4.0, 2.0, 1.5],
            velocity=[0.0, 0.0],
        )
        self.assertIsNone(
            unpack_objects([row], 100000).tracked_objects.tracked_objects[0].angular_velocity
        )


if __name__ == "__main__":
    unittest.main()
