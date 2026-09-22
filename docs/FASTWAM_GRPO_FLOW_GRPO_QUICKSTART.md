# Fast-WAM + RoboDojo quickstart

This is the shortest supported path for running native evaluation, GRFPO, or
Flow-GRPO on another Slurm cluster. Machine paths belong in an external site
file or experiment input; do not edit the framework code to install it.

## 1. Prepare the checkouts and Python environment

Keep the projects as separate source trees:

```text
<workspace>/verl-vla
<workspace>/RoboDojo
<workspace>/RoboDojo/XPolicyLab/policy/FastWAM/FastWAM
```

Install the compatible RoboDojo/Isaac, Fast-WAM, PyTorch/CUDA, verl, and
verl-vla dependencies in one Python environment. Then copy the site template
outside the repository and replace every placeholder:

```bash
cd <workspace>/verl-vla
cp configs/site/example.env.sh /path/to/my-site.env.sh
editor /path/to/my-site.env.sh
source /path/to/my-site.env.sh
```

Validate that the selected interpreter can import the complete runtime:

```bash
"$VVLA_PYTHON" scripts/lib/check_runtime.py \
  --repo-root "$PWD" \
  --robodojo-root "$ROBODOJO_ROOT" \
  --policy-root "$FASTWAM_POLICY_ROOT"
```

## 2. Restore the selected SFT checkpoint

Download these files from the OneDrive `robodojo_official5_job554858`
directory:

```text
weights/step_020412.pt
_archive_metadata/dataset_stats.json
```

Restore them under any local model root using the upstream Fast-WAM filename:

```text
/models/sft_c20412/
├── checkpoints/weights/step_020000.pt  # contents of step_020412.pt
└── dataset_stats.json
```

Verify the files:

```bash
sha256sum \
  /models/sft_c20412/checkpoints/weights/step_020000.pt \
  /models/sft_c20412/dataset_stats.json
```

The expected hashes are:

```text
98621f8b068ed4adf7a799e6af0d712cd4ccad3b9b0c724c95796e4fe3ca3488  step_020000.pt
af119307e4fdd0cc0c052e2017d1aa2afc270a07e7d0dedf1f325b86f3c8c4ef  dataset_stats.json
```

Create a target-machine selection receipt, for example
`/models/sft_c20412/selection.json`:

```json
{
  "selection_protocol": "native_fastwam_robodojo_full_observation_multitask5_v2",
  "eval_checkpoint_root": "/models/sft_c20412",
  "checkpoint_sha256": "98621f8b068ed4adf7a799e6af0d712cd4ccad3b9b0c724c95796e4fe3ca3488",
  "selected": {
    "label": "sft_c20412",
    "step": 20412,
    "source_checkpoint": "/models/sft_c20412/checkpoints/weights/step_020000.pt"
  }
}
```

Set `VVLA_MULTITASK_SFT_SELECTION` in the site file to this receipt, or pass
it explicitly with `--selection`.

## 3. Run native evaluation

Copy the example and edit its policy, task, layout, seed, and topology fields:

```bash
cp configs/eval/fastwam/portable_example.json /path/to/eval.json
editor /path/to/eval.json
```

For the restored SFT policy, set `model_root` to `/models/sft_c20412`, use the
weight SHA-256 above, and set both `verl_actor_checkpoint` and
`expected_source_verl_global_step` to `null`. Paths inside the spec may be
relative to the spec file.

Validate the resolved job without submitting it:

```bash
"$VVLA_PYTHON" scripts/eval/mainline/eval.py launch \
  --spec /path/to/eval.json \
  --output-root "$VVLA_OUTPUT_ROOT/eval/sft_c20412" \
  --policy-root "$FASTWAM_POLICY_ROOT" \
  --robodojo-root "$ROBODOJO_ROOT" \
  --hf-home "$HF_HOME" \
  --python "$VVLA_PYTHON" \
  --native-lib-dir "$VVLA_NATIVE_LIB_DIR" \
  --partition normal \
  --lanes 1 \
  --coalesced-items-per-evaluator 1 \
  --dry-run
```

Remove `--dry-run` to submit. Monitor and summarize the frozen run manifest:

```bash
"$VVLA_PYTHON" scripts/eval/mainline/eval.py status \
  --run-manifest "$VVLA_OUTPUT_ROOT/eval/sft_c20412/RUN_MANIFEST.json"

"$VVLA_PYTHON" scripts/eval/mainline/eval.py summarize \
  --run-manifest "$VVLA_OUTPUT_ROOT/eval/sft_c20412/RUN_MANIFEST.json"
```

## 4. Run GRFPO

Always start with a dry run:

```bash
scripts/grfpo/mainline/launch.sh \
  --run-id multitask5_grfpo \
  --selection /models/sft_c20412/selection.json \
  --updates 8 \
  --minimum-accepted-groups 32 \
  --candidate-groups 96 \
  --process-score \
  --task-equal \
  --collector-count 12 \
  --collector-concurrency 8 \
  --simulators-per-gpu 1 \
  --train-gpus-per-node 2 \
  --train-policy-microbatch 4 \
  --train-fsdp-size 1 \
  --actor-lr 5e-6 \
  --dry-run
```

Inspect the resolved runtime and Slurm command, then remove `--dry-run` to
submit. The default multitask schedule has four tasks and three suites, so its
initial wave requires 12 collector lanes. Increase `--simulators-per-gpu` only
after validating that topology on the target cluster.

Manage a submitted run with:

```bash
scripts/grfpo/mainline/manage.sh status multitask5_grfpo
scripts/grfpo/mainline/manage.sh resume multitask5_grfpo
```

## 5. Run Flow-GRPO

Flow-GRPO uses the same collection and durable checkpoint chain. Use the Flow
wrapper and add its method-specific settings:

```bash
scripts/flow_grpo/mainline/launch.sh \
  --run-id multitask5_flow_grpo \
  --selection /models/sft_c20412/selection.json \
  --updates 8 \
  --minimum-accepted-groups 32 \
  --candidate-groups 96 \
  --process-score \
  --task-equal \
  --collector-count 12 \
  --collector-concurrency 8 \
  --simulators-per-gpu 1 \
  --train-gpus-per-node 2 \
  --train-policy-microbatch 4 \
  --train-fsdp-size 1 \
  --actor-lr 5e-6 \
  --flow-grpo-noise-level 0.01 \
  --flow-grpo-transition-batch-size 3 \
  --dry-run
```

Remove `--dry-run` only after checking the resolved manifest. Flow-GRPO output
is stored under the `flow_grpo/` namespace instead of `grfpo/`.

## 6. Important constraints

- The maintained launchers currently require Slurm.
- RoboDojo and Fast-WAM remain external source dependencies.
- Keep `defer_chunk_observations=false`; the faster deferred path changes the
  RoboDojo render/sensor cadence and is not evaluation-equivalent.
- A verl actor checkpoint is required only when evaluating or resuming an RL
  policy. The selected SFT checkpoint is already in native Fast-WAM format.
- Do not commit checkpoints, datasets, output directories, site files, or
  machine-specific selection receipts.

For the full runtime and topology explanation, see
`docs/PORTABLE_FASTWAM_ROBODOJO_WORKFLOWS.md`.
