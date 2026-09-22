# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import os
import queue
import subprocess
import time
from typing import Optional

import torch
import torch.multiprocessing as mp
from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)


class SimulatorCommandTimeoutError(TimeoutError):
    """Raised when a live simulator subprocess does not answer an RPC in time."""


def cleanup_device_tensors():
    gc.collect()
    torch_device = get_torch_device()
    empty_cache = getattr(torch_device, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def get_gpu_numa_node(gpu_id: int) -> int:
    try:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            # Get PCI bus info
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId
        except ImportError:
            # Fallback to nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            pci_bus_id = result.stdout.strip()

        # Extract bus number from PCI bus ID (format: 0000:XX:YY.Z)
        bus_number = pci_bus_id.split(":")[1]

        # Get NUMA node from sysfs
        numa_node_path = f"/sys/bus/pci/devices/0000:{bus_number}:00.0/numa_node"
        if os.path.exists(numa_node_path):
            with open(numa_node_path) as f:
                numa_node = int(f.read().strip())
                if numa_node >= 0:
                    return numa_node

        # Fallback: try to get from lscpu
        result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
        numa_nodes = 0
        for line in result.stdout.split("\n"):
            if "NUMA node(s):" in line:
                numa_nodes = int(line.split(":")[1].strip())
                break

        # If we can't determine the exact NUMA node, distribute evenly
        return gpu_id % numa_nodes if numa_nodes > 0 else 0

    except Exception as e:
        logger.error(f"Warning: Could not determine NUMA node for GPU {gpu_id}: {e}")
        return 0


def get_numa_cpus(numa_node: int) -> list:
    try:
        # Read from sysfs
        cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        if os.path.exists(cpulist_path):
            with open(cpulist_path) as f:
                cpulist = f.read().strip()

            # Parse CPU list (e.g., "0-7,16-23" or "0,1,2,3")
            cpus = []
            for part in cpulist.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    cpus.extend(range(start, end + 1))
                else:
                    cpus.append(int(part))
            return cpus
    except Exception as e:
        logger.error(f"Warning: Could not get CPU list for NUMA node {numa_node}: {e}")

    # Fallback: return all available CPUs
    return list(range(os.cpu_count() or 1))


def set_process_numa_affinity(gpu_id: int) -> None:
    try:
        numa_node = get_gpu_numa_node(gpu_id)
        cpus = get_numa_cpus(numa_node)

        if not cpus:
            logger.error(f"Warning: No CPUs found for NUMA node {numa_node}")
            return

        os.sched_setaffinity(0, cpus)
        try:
            subprocess.run(
                ["numactl", "--membind", str(numa_node), "--"],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass  # numactl not available, that's ok

    except Exception as e:
        logger.error(f"Warning: Could not set NUMA affinity for GPU {gpu_id}: {e}")


def recursive_to_own(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_shared() else obj
    elif isinstance(obj, list):
        return [recursive_to_own(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_own(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_own(v) for k, v in obj.items()}
    else:
        return obj


class EnvManager:
    def __init__(
        self,
        cfg,
        rank,
        world_size,
        env_cls,
        stage_id: int = 0,
        stage_num: int = 1,
        only_eval: bool = False,
        start_timeout_s: int = 180,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.stage_id = stage_id
        self.stage_num = stage_num
        self.only_eval = only_eval
        self.start_timeout_s = int(start_timeout_s)
        self.process: Optional[mp.Process] = None
        self.command_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.state_buffer: Optional[bytes] = None
        self.cold_start_pending = False

        self.env_cls = env_cls

    def start_simulator(self):
        """Start simulator process with shared memory queues"""
        if self.process:
            logger.info(f"Simulator process already running for rank {self.rank}")
            return

        self.context = mp.get_context("spawn")
        # Create shared memory queues
        self.command_queue = self.context.Queue()
        self.result_queue = self.context.Queue()

        # Start simulator process
        self.process = self.context.Process(
            target=_simulator_worker,
            args=(
                self.cfg,
                self.rank,
                self.world_size,
                self.stage_id,
                self.stage_num,
                self.env_cls,
                self.only_eval,
                self.command_queue,
                self.result_queue,
                self.state_buffer,
                torch.cuda.is_available(),
            ),
        )
        self.process.start()

        # Wait for initialization, but fail fast if the subprocess dies before it
        # can report either "ready" or "error".
        deadline = time.monotonic() + self.start_timeout_s
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                exitcode = self.process.exitcode if self.process is not None else None
                alive = self.process.is_alive() if self.process is not None else False
                raise RuntimeError(
                    "Simulator initialization timed out "
                    f"after {self.start_timeout_s}s "
                    f"(rank={self.rank}, stage_id={self.stage_id}, env_cls={self.env_cls.__name__}, "
                    f"process_alive={alive}, exitcode={exitcode})."
                )

            try:
                result = self.result_queue.get(timeout=min(1.0, remaining_s))
                break
            except queue.Empty:
                if self.process is not None and not self.process.is_alive():
                    raise RuntimeError(
                        "Simulator initialization process exited before reporting ready "
                        f"(rank={self.rank}, stage_id={self.stage_id}, env_cls={self.env_cls.__name__}, "
                        f"pid={self.process.pid}, exitcode={self.process.exitcode})."
                    ) from None
        if result["status"] != "ready":
            raise RuntimeError(f"Simulator initialization failed: {result}")
        # RoboDojo lazily builds the actual scene during reset.  Track that
        # boundary in the parent so EnvWorker can stagger only expensive cold
        # resets, without serializing normal steady-state resets.
        self.cold_start_pending = True

    def stop_simulator(self):
        if not self.process:
            return

        # Request state save
        self.command_queue.put({"method": "get_state", "args": [], "kwargs": {}})

        # Get saved state
        result = self.result_queue.get(timeout=180)
        if result["status"] == "success":
            self.state_buffer = result["data"]

        self.command_queue.put({"method": "shutdown"})
        self.command_queue.close()
        self.result_queue.close()
        self.command_queue = None
        self.result_queue = None
        self.process.join(timeout=5)

        self.command_queue = None
        self.result_queue = None
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.process = None

    def force_stop_simulator(self) -> None:
        """Stop an unresponsive subprocess without requesting adapter state.

        ``state_buffer`` is deliberately preserved.  A normal restart saves the
        deterministic case cursors before launching the child; if a later reset
        RPC hangs, the replacement child must reload that last known-good state
        and retry the same case rather than advancing the schedule.
        """

        process = self.process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                process.join(timeout=10)

        for command_queue in (self.command_queue, self.result_queue):
            if command_queue is None:
                continue
            try:
                command_queue.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                command_queue.close()
            except (OSError, ValueError):
                pass

        self.command_queue = None
        self.result_queue = None
        self.process = None

    def call(self, name, *args, timeout_s: float | None = None, **kwargs):
        """Invoke one simulator method, optionally with a bounded response wait."""

        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Simulator not running")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive when set, got {timeout_s}")

        args = recursive_to_own(args)
        kwargs = recursive_to_own(kwargs)
        self.command_queue.put({"method": name, "args": args, "kwargs": kwargs})

        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            wait_s = 1.0
            if deadline is not None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise SimulatorCommandTimeoutError(
                        f"Simulator RPC {name!r} timed out after {timeout_s}s "
                        f"(rank={self.rank}, stage_id={self.stage_id}, pid={self.process.pid})."
                    )
                wait_s = min(wait_s, remaining_s)
            try:
                result = self.result_queue.get(timeout=wait_s)
                break
            except queue.Empty:
                if self.process is None or not self.process.is_alive():
                    exitcode = None if self.process is None else self.process.exitcode
                    raise RuntimeError(
                        f"Simulator exited while handling RPC {name!r} "
                        f"(rank={self.rank}, stage_id={self.stage_id}, exitcode={exitcode})."
                    ) from None

        result = recursive_to_own(result)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        if name == "reset":
            self.cold_start_pending = False
        return result["data"]

    def snapshot_state(self, *, timeout_s: float = 30.0):
        """Persist the latest adapter state for a future forced restart.

        A normal simulator restart snapshots state before shutting down.  A
        recovery restart, however, can happen after several successful reset
        calls (notably while switching between train and evaluation).  Keep
        the recovery point current after those calls so a later timeout cannot
        roll deterministic case cursors back to an older reset wave.
        """

        state = self.call("get_state", timeout_s=timeout_s)
        self.state_buffer = state
        return state

    def __getattr__(self, name):
        if name in [
            "cfg",
            "rank",
            "world_size",
            "stage_id",
            "stage_num",
            "only_eval",
            "process",
            "command_queue",
            "result_queue",
            "state_buffer",
            "env_cls",
            "context",
            "start_timeout_s",
            "cold_start_pending",
        ]:
            return super().__getattr__(name)

        def method_proxy(*args, **kwargs):
            return self.call(name, *args, **kwargs)

        return method_proxy

    def __setattr__(self, name, value):
        # Handle special attributes that should be set on self
        if name in [
            "cfg",
            "rank",
            "world_size",
            "stage_id",
            "stage_num",
            "only_eval",
            "process",
            "command_queue",
            "result_queue",
            "state_buffer",
            "env_cls",
            "context",
            "start_timeout_s",
            "cold_start_pending",
        ]:
            super().__setattr__(name, value)
            return

        if self.process is None or not self.process.is_alive():
            raise RuntimeError(f"Simulator not running to set attribute {name} to {value}")

        value = recursive_to_own(value)
        self.command_queue.put(
            {
                "method": "__setattr__",
                "args": [name, value],
                "kwargs": {},
            }
        )

        result = self.result_queue.get()
        result = recursive_to_own(result)
        if result["status"] == "error":
            raise Exception(result["error"])


def _simulator_worker(
    cfg,
    rank,
    world_size,
    stage_id,
    stage_num,
    env_cls,
    only_eval,
    command_queue,
    result_queue,
    state_buffer,
    bind_numa=True,
):
    """Worker process for simulator"""
    # Set NUMA affinity for the process to match the GPU rank
    import logging
    import os

    pid = os.getpid()
    logger = logging.getLogger(f"simulator_worker_{rank}_{pid}")

    if bind_numa:
        set_process_numa_affinity(rank)
    try:
        try:
            env = env_cls(cfg, rank, world_size, stage_id=stage_id, stage_num=stage_num, only_eval=only_eval)
        except TypeError:
            env = env_cls(cfg, rank, world_size)

        if state_buffer:
            env.load_state(state_buffer)

        # Signal ready
        result_queue.put({"status": "ready"})

        # Main command processing loop
        while True:
            try:
                command = command_queue.get()
                logger.debug(f"Received command method: {command['method']}")

                if command["method"] == "shutdown":
                    env.close()
                    break

                method_name = command["method"]
                args = command.get("args", [])
                kwargs = command.get("kwargs", {})
                if method_name == "__setattr__":
                    # Handle attribute setting
                    attr_name, attr_value = args
                    setattr(env, attr_name, attr_value)
                    result_queue.put({"status": "success", "data": None})
                elif hasattr(env, method_name):
                    method = getattr(env, method_name)
                    assert callable(method), f"Method {method_name} is not callable"
                    result = method(*args, **kwargs)
                    result_queue.put({"status": "success", "data": result})
                else:
                    logger.error(f"Method '{method_name}' not found")
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"Method '{method_name}' not found",
                        }
                    )

            except Exception as e:
                logger.exception(e)
                result_queue.put({"status": "error", "error": str(e)})

    except Exception as e:
        logger.exception(e)
        result_queue.put({"status": "error", "error": str(e)})

    finally:
        command_queue.close()
        result_queue.close()
