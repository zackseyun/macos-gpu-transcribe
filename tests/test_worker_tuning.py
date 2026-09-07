import unittest
from pathlib import Path
from unittest import mock

import huggingface_hub

import transcribe_worker


class CohereModelDirTest(unittest.TestCase):
    def test_uses_cached_snapshot_without_network(self):
        calls = []

        def fake_snapshot_download(repo_id, allow_patterns, local_files_only=False):
            calls.append(local_files_only)
            return "/cache/snapshot"

        with mock.patch.object(huggingface_hub, "snapshot_download", fake_snapshot_download):
            path = transcribe_worker._resolve_cohere_mlx_model_dir()

        self.assertEqual([True], calls)
        self.assertEqual(Path("/cache/snapshot") / transcribe_worker.COHERE_MLX_SUBDIR, path)

    def test_falls_back_to_network_when_not_cached(self):
        calls = []

        def fake_snapshot_download(repo_id, allow_patterns, local_files_only=False):
            calls.append(local_files_only)
            if local_files_only:
                raise FileNotFoundError("not cached yet")
            return "/cache/snapshot"

        with mock.patch.object(huggingface_hub, "snapshot_download", fake_snapshot_download):
            path = transcribe_worker._resolve_cohere_mlx_model_dir()

        self.assertEqual([True, False], calls)
        self.assertEqual(Path("/cache/snapshot") / transcribe_worker.COHERE_MLX_SUBDIR, path)


class WarmCadenceTest(unittest.TestCase):
    def test_on_demand_warm_skip_outlasts_keep_warm_cadence(self):
        # Otherwise a Fn press right before the next keep-warm tick launches a
        # redundant warm that holds inference_lock while the real clip waits.
        self.assertGreater(
            transcribe_worker.ON_DEMAND_WARM_SKIP_THRESHOLD,
            transcribe_worker.KEEP_WARM_INTERVAL + 2 * transcribe_worker.KEEP_WARM_CHECK_INTERVAL,
        )

    def test_mlx_memory_limits_are_bounded(self):
        gib = 1024 ** 3
        # Scratch peaks near 1.4GB per 35s chunk; the cache must hold that but
        # must not recreate the old ~10GB swapped-out footprint.
        self.assertGreaterEqual(transcribe_worker.METAL_CACHE_LIMIT_BYTES, int(1.5 * gib))
        self.assertLessEqual(transcribe_worker.METAL_CACHE_LIMIT_BYTES, 3 * gib)
        self.assertGreaterEqual(transcribe_worker.METAL_WIRED_LIMIT_BYTES, 6 * gib)


if __name__ == "__main__":
    unittest.main()
