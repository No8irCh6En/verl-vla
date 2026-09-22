from types import SimpleNamespace

import numpy as np
import torch

from verl_vla.workflows.native_fastwam_robodojo_eval import (
    infer_shared_model_in_microbatches,
    infer_shared_model_outputs_in_microbatches,
    merge_ipc_observations,
    terminal_or_latest_scores,
)


class _SharedModel:
    def __init__(self):
        self.batch_sizes = []

    def sac_sample_actions(self, batch, eval):
        assert eval is True
        size = len(batch)
        self.batch_sizes.append(size)
        policy_seeds = torch.as_tensor(batch.non_tensor_batch["policy_seed"])
        action = policy_seeds[:, None, None].expand(size, 24, 14).to(torch.float32)
        return SimpleNamespace(action=action)


class _Output:
    def __init__(self, action):
        self.action = action

    def to_data_proto(self):
        from verl import DataProto

        full_action = torch.nn.functional.pad(self.action, (0, 0, 0, 8))
        return DataProto.from_dict(tensors={"action": self.action, "full_action": full_action})


class _CanonicalSharedModel(_SharedModel):
    def sac_sample_actions(self, batch, eval):
        result = super().sac_sample_actions(batch, eval)
        return _Output(result.action)


class _FlowTraceOutput(_Output):
    def to_data_proto(self):
        output = super().to_data_proto()
        batch_size = output.batch["action"].shape[0]
        output.batch["flow_grpo.latents"] = torch.zeros(batch_size, 3, 32, 14)
        output.batch["flow_grpo.old_log_probs"] = torch.zeros(batch_size, 2, 32)
        output.batch["flow_grpo.sigmas"] = torch.zeros(batch_size, 2)
        output.batch["flow_grpo.deltas"] = torch.ones(batch_size, 2)
        return output


class _FlowSharedModel:
    def __init__(self):
        self.eval_values = []

    def sac_sample_actions(self, batch, eval):
        self.eval_values.append(eval)
        size = len(batch)
        return _FlowTraceOutput(torch.zeros(size, 24, 14))


def test_one_shared_model_processes_vector_batch_in_microbatches():
    obs = {
        "observation": [
            {
                "observation.images.cam_head": np.zeros((240, 320, 3), dtype=np.uint8),
                "observation.images.cam_left_wrist": np.zeros((240, 320, 3), dtype=np.uint8),
                "observation.images.cam_right_wrist": np.zeros((240, 320, 3), dtype=np.uint8),
                "observation.state": np.zeros(14, dtype=np.float32),
            }
            for _ in range(10)
        ],
        "task": ["stack bowls"] * 10,
        "policy_seed": np.arange(10, dtype=np.int64),
    }
    model = _SharedModel()

    actions = infer_shared_model_in_microbatches(model, obs, microbatch_size=4)

    assert model.batch_sizes == [4, 4, 2]
    assert actions.shape == (10, 24, 14)
    torch.testing.assert_close(actions[:, 0, 0], torch.arange(10, dtype=torch.float32))


def test_canonical_microbatch_output_keeps_executed_and_full_action():
    obs = {
        "observation": [
            {
                "observation.images.cam_head": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation.state": np.zeros(14, dtype=np.float32),
            }
            for _ in range(8)
        ],
        "task": ["stack bowls"] * 8,
        "policy_seed": np.arange(8, dtype=np.int64),
    }
    output = infer_shared_model_outputs_in_microbatches(_CanonicalSharedModel(), obs, microbatch_size=3)
    assert output.batch["action"].shape == (8, 24, 14)
    assert output.batch["full_action"].shape == (8, 32, 14)


def test_flow_collection_keeps_sde_trace_and_disables_deterministic_eval_mode():
    obs = {
        "observation": [
            {
                "observation.images.cam_head": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation.state": np.zeros(14, dtype=np.float32),
            }
            for _ in range(8)
        ],
        "task": ["stack bowls"] * 8,
        "policy_seed": np.arange(8, dtype=np.int64),
    }
    model = _FlowSharedModel()
    output = infer_shared_model_outputs_in_microbatches(model, obs, microbatch_size=3, eval=False)
    assert model.eval_values == [False, False, False]
    assert output.batch["flow_grpo.latents"].shape == (8, 3, 32, 14)
    assert output.batch["flow_grpo.old_log_probs"].shape == (8, 2, 32)
    assert output.batch["flow_grpo.sigmas"].shape == (8, 2)
    assert output.batch["flow_grpo.deltas"].shape == (8, 2)


def test_shared_simulator_requests_merge_and_keep_exact_output_slices():
    def request(offset: int):
        return {
            "observation": [
                {
                    "observation.images.cam_head": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation.state": np.full(14, offset + index, dtype=np.float32),
                }
                for index in range(8)
            ],
            "task": [f"task-{offset}"] * 8,
            "task_id": np.full(8, offset, dtype=np.int64),
            "policy_seed": np.arange(offset, offset + 8, dtype=np.int64),
        }

    merged, slices = merge_ipc_observations([request(100), request(200)])

    assert slices == [slice(0, 8), slice(8, 16)]
    assert len(merged["observation"]) == len(merged["task"]) == 16
    np.testing.assert_array_equal(merged["policy_seed"][slices[0]], np.arange(100, 108))
    np.testing.assert_array_equal(merged["policy_seed"][slices[1]], np.arange(200, 208))
    assert merged["task"][:8] == ["task-100"] * 8
    assert merged["task"][8:] == ["task-200"] * 8


def test_terminal_score_uses_first_done_not_chunk_max_or_zero_padding():
    scores = torch.tensor(
        [
            [0.0, 0.15, 1.0, 0.0],
            [0.0, 0.15, 0.10, 0.05],
            [0.9, 0.8, 0.7, 0.6],
        ]
    )
    terminated = torch.tensor(
        [
            [False, False, True, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    truncated = torch.zeros_like(terminated)

    result = terminal_or_latest_scores(
        scores,
        terminated,
        truncated,
        already_done=np.asarray([False, False, True]),
    )

    np.testing.assert_allclose(result, [1.0, 0.05, 0.0])
