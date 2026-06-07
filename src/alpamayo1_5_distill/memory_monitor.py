# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU memory monitoring and automatic cleanup for distillation training.

Provides real-time memory tracking, warning alerts, and automatic cache cleanup
to prevent OOM errors during long training runs.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """GPU memory monitor with automatic cleanup.

    Args:
        device: CUDA device (e.g., "cuda:0" or "cuda").
        warning_threshold: Fraction of total memory to trigger warning (default: 0.85).
        cleanup_threshold: Fraction of total memory to trigger cleanup (default: 0.90).
    """

    def __init__(
        self,
        device: str = "cuda",
        warning_threshold: float = 0.85,
        cleanup_threshold: float = 0.90,
    ):
        self.device = device
        self.warning_threshold = warning_threshold
        self.cleanup_threshold = cleanup_threshold

    def get_stats(self) -> dict[str, int]:
        """Get current GPU memory statistics in MiB.

        Returns:
            Dict with keys: allocated, reserved, total, free
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(self.device) // (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) // (1024 ** 2)
        total = torch.cuda.get_device_properties(self.device).total_memory // (1024 ** 2)
        free = total - reserved

        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": free,
        }

    def check_and_cleanup(self, step: int) -> bool:
        """Check memory status and trigger cleanup if needed.

        Args:
            step: Current training step.

        Returns:
            True if cleanup was triggered.
        """
        stats = self.get_stats()
        usage_ratio = stats["reserved"] / stats["total"] if stats["total"] > 0 else 0

        # Log memory stats periodically
        if step % 100 == 0:
            logger.info(
                "GPU Memory | Step %d | Allocated: %d MiB | Reserved: %d MiB | "
                "Free: %d MiB | Total: %d MiB | Usage: %.1f%%",
                step, stats["allocated"], stats["reserved"], stats["free"], stats["total"],
                usage_ratio * 100,
            )

        # Warning threshold
        if usage_ratio > self.warning_threshold:
            logger.warning(
                "GPU Memory HIGH | Step %d | Usage: %.1f%% | Reserved: %d MiB | Free: %d MiB",
                step, usage_ratio * 100, stats["reserved"], stats["free"],
            )

        # Cleanup threshold
        if usage_ratio > self.cleanup_threshold:
            logger.warning(
                "GPU Memory CRITICAL | Step %d | Usage: %.1f%% | Triggering cleanup",
                step, usage_ratio * 100,
            )
            self.cleanup()
            return True

        return False

    def cleanup(self):
        """Free GPU memory by emptying cache and running garbage collection."""
        if not torch.cuda.is_available():
            return

        # Clear PyTorch cache
        torch.cuda.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        stats = self.get_stats()
        logger.info(
            "Memory cleanup | Allocated: %d MiB | Reserved: %d MiB | Free: %d MiB",
            stats["allocated"], stats["reserved"], stats["free"],
        )
