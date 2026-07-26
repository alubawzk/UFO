from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import humanoidverse.distributed as distributed


class DistributedControlBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        distributed._CONTROL_GROUP = None

    def tearDown(self) -> None:
        distributed._CONTROL_GROUP = None

    @patch.object(distributed.dist, "new_group")
    @patch.object(distributed.dist, "get_backend", return_value="nccl")
    @patch.object(distributed.dist, "get_world_size", return_value=8)
    @patch.object(distributed.dist, "is_initialized", return_value=True)
    @patch.object(distributed.dist, "is_available", return_value=True)
    def test_nccl_training_creates_gloo_control_group(
        self,
        _is_available: Mock,
        _is_initialized: Mock,
        _get_world_size: Mock,
        _get_backend: Mock,
        new_group: Mock,
    ) -> None:
        group = object()
        new_group.return_value = group

        distributed.initialize_control_group()

        self.assertIs(distributed._CONTROL_GROUP, group)
        self.assertEqual(new_group.call_args.kwargs["backend"], "gloo")

    @patch.object(distributed.dist, "barrier")
    @patch.object(distributed.dist, "get_backend", return_value="nccl")
    @patch.object(distributed.dist, "get_world_size", return_value=8)
    @patch.object(distributed.dist, "is_initialized", return_value=True)
    @patch.object(distributed.dist, "is_available", return_value=True)
    def test_control_barrier_uses_gloo_group(
        self,
        _is_available: Mock,
        _is_initialized: Mock,
        _get_world_size: Mock,
        _get_backend: Mock,
        barrier: Mock,
    ) -> None:
        group = object()
        distributed._CONTROL_GROUP = group

        distributed.control_barrier()

        barrier.assert_called_once_with(group=group)


if __name__ == "__main__":
    unittest.main()
