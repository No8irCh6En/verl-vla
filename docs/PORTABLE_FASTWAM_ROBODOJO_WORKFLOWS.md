# Portable Fast-WAM + RoboDojo workflows

This document describes the supported portable path for GRFPO rollout/training
and native Fast-WAM + RoboDojo evaluation. Historical one-off recovery and
review scripts are not public entrypoints.

## 1. Runtime layout

The three source trees remain separate:

```text
<workspace>/verl-vla
<workspace>/RoboDojo
<workspace>/RoboDojo/XPolicyLab/policy/FastWAM/FastWAM
```

Do not copy RoboDojo or Fast-WAM into verl-vla. The launcher resolves their
locations once and freezes the resolved paths in each experiment/run manifest.

## 2. Python environment

The selected Python environment must import all runtime packages used by the
combined workflow. Run the same fail-fast check used by both public launchers:

```bash
"$VVLA_PYTHON" scripts/lib/check_runtime.py \
  --repo-root "$VVLA_REPO_ROOT" \
  --robodojo-root "$ROBODOJO_ROOT" \
  --policy-root "$FASTWAM_POLICY_ROOT"
```

Install verl-vla and Fast-WAM from their project metadata. Fast-WAM declares
`GitPython`, `accelerate`, and `termcolor`; an environment assembled by manually
combining site-packages can miss transitive imports even when the Fast-WAM
source checkout is present. The preflight imports `fastwam.runtime` itself, so
these failures happen before any GPU job is submitted.

Keep the virtual-environment `bin/python` entrypoint. Do not replace it with
the result of `readlink -f`: an overlay environment may symlink to a base
interpreter but select different site-packages through the original entrypoint.

## 3. Site configuration

Copy and edit the example without committing machine paths:

```bash
cp configs/site/example.env.sh /path/outside/the/repo/my-site.env.sh
editor /path/outside/the/repo/my-site.env.sh
source /path/outside/the/repo/my-site.env.sh
```

Required runtime values are:

- `ROBODOJO_ROOT`
- `FASTWAM_POLICY_ROOT`
- `VVLA_PYTHON`
- `VVLA_OUTPUT_ROOT`
- `HF_HOME`
- the SFT selection receipt, passed as `--selection` or
  `VVLA_MULTITASK_SFT_SELECTION`

The Slurm account, partition, QoS, module, and node exclude list belong only in
the site file. Algorithm YAML and public launchers do not contain Peilab paths.

## 4. GRFPO and Flow-GRPO rollout + train

The only supported user entry is:

```bash
scripts/grfpo/mainline/launch.sh \
  --run-id my_grfpo_run \
  --selection /path/to/sft_selection.json \
  --updates 8 \
  --minimum-accepted-groups 32 \
  --candidate-groups 96 \
  --process-score \
  --task-equal \
  --collector-count 12 \
  --collector-concurrency 8 \
  --simulators-per-gpu 3 \
  --train-gpus-per-node 2 \
  --train-policy-microbatch 4 \
  --train-fsdp-size 1 \
  --actor-lr 5e-6
```

Flow-GRPO uses the same durable collection and training chain. Invoke
`scripts/flow_grpo/mainline/launch.sh` with the same arguments; the wrapper
selects the Flow-GRPO actor objective and writes under the `flow_grpo/`
namespace. See `docs/FASTWAM_GRPO_FLOW_GRPO_QUICKSTART.md` for a complete
command.

Run the same command with `--dry-run` first. It may be invoked from any current
working directory. The preview must show a `resolved_runtime` section and the
intended Slurm command, but it does not submit a job or initialize a run.

Manage a submitted run with:

```bash
scripts/grfpo/mainline/manage.sh status RUN_ID
scripts/grfpo/mainline/manage.sh replace-slow RUN_ID LANE reason
scripts/grfpo/mainline/manage.sh add-collectors RUN_ID COUNT preempt
scripts/grfpo/mainline/manage.sh resume RUN_ID
```

The chain passes one immutable runtime contract through the controller,
collectors, watcher, and actor trainer. Workers no longer import from the old
`progress/phase6b/runtime_site` directory.

The production collector count is structural, not a throughput knob: the
initial warm-start wave requires one lane per configured `(task, suite)`. The
default four-task, three-suite run therefore requires `--collector-count 12`.
Use `--collector-concurrency` and opportunistic collectors to control resource
placement. The launcher rejects an inconsistent lane count before submitting a
GPU job.

## 5. Native evaluation

An eval spec owns policy identities and fixed conditions. Start by copying
`configs/eval/fastwam/portable_example.json`. Paths inside a spec are resolved
relative to the spec file; external runtime roots come from the site contract
or explicit arguments. Replace the placeholder SHA-256 with the immutable SFT
checkpoint checksum used by that policy.

```bash
"$VVLA_PYTHON" scripts/eval/mainline/eval.py launch \
  --spec /path/to/eval_spec.json \
  --output-root "$VVLA_OUTPUT_ROOT/eval/my_run" \
  --policy-root "$FASTWAM_POLICY_ROOT" \
  --robodojo-root "$ROBODOJO_ROOT" \
  --hf-home "$HF_HOME" \
  --python "$VVLA_PYTHON" \
  --native-lib-dir "$VVLA_NATIVE_LIB_DIR" \
  --partition normal \
  --lanes 8 \
  --coalesced-items-per-evaluator 3 \
  --dry-run
```

Remove `--dry-run` to submit. The resulting `RUN_MANIFEST.json` freezes every
resolved path and the site resource contract. Repair lanes read this manifest;
they do not contain a second copy of machine-specific paths.

For a split normal/preempt run, first register the manifest with `eval.py
launch --dry-run`, then use:

```bash
scripts/eval/mainline/launch_split_after_complete.sh \
  /path/to/ready-marker \
  /path/to/RUN_MANIFEST.json \
  /path/to/eval-output \
  4 4 12:00:00 1 3
```

The last two values select one shared Fast-WAM evaluator per GPU and three
RoboDojo vecN shards served by that model.

## 6. Portability checks

Run the focused CPU tests before publishing or moving the checkout:

```bash
"$VVLA_PYTHON" -m pytest -q \
  tests/scripts/test_portable_mainline_paths.py \
  tests/scripts/test_fixed_topology_eval.py \
  tests/scripts/test_split_eval_monitor.py \
  tests/trainer/test_grfpo_launch_config.py \
  tests/workflows/test_native_fastwam_robodojo_eval.py
```

These cover the supported-runtime hardcode boundary, Python overlay symlink
preservation, eval topology/manifest behavior, and GRFPO launch semantics.

On 2026-09-22 the same paths were also exercised on real Slurm GPUs: one G=8
rollout was atomically committed under
`outputs/grfpo/portability_rollout_worker_smoke_20260922`, and one complete
288-trajectory, video-enabled native evaluation was finalized under
`outputs/eval/process_task_equal_reward_control_reverse_vec6_288_v4/theta6`.
These prove the current site installation end to end; a different cluster must
still perform its own minimal GPU smoke for CUDA/Isaac binary compatibility.

## 7. Scope

The portable public surface is the mainline GRFPO, Flow-GRPO, and native-eval
workflows above. Old `phase*`, reviewer-specific, checkpoint-recovery, storage,
and archived evaluation scripts may still record original experiment paths;
they are evidence/operations history and are not imported by these entrypoints.

GPU behavior has been validated on the current Peilab installation. A new
cluster still needs its own Isaac/RoboDojo installation and a small smoke run;
path portability does not claim binary compatibility across driver, CUDA, or
Isaac versions.

The supported portable surface is currently GRFPO and Flow-GRPO
rollout/training plus native Fast-WAM/RoboDojo evaluation. SFT training,
archival utilities, and historical experiment scripts are not covered by this
contract and must not be used as evidence that the mainline still contains
hard-coded paths.
