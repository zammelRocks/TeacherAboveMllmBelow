"""Regression test for a real bug found while investigating a zipfile
"Duplicate name" warning during a live train-surrogate run: EmbeddingCache
used str(Path(...).resolve()) as its key, which is backslash-separated on
Windows, but np.savez/np.load silently store/report member names with
forward slashes (zipfile.ZipInfo converts os.sep -> "/"). After any
save-then-reload cycle, every fresh key (backslash) failed to match every
reloaded key (forward-slash), so every lookup "missed", silently
recomputed, and re-inserted a duplicate entry under the mismatched key --
which then collided on the next save. Confirmed empirically: no embedding
values were corrupted (recomputation is deterministic), but caching across
a save/reload was completely defeated, and the cache file grew ~2x on
every reuse.

This test doesn't just check "the array value round-trips correctly" (that
already passed before the fix); it checks that a reload is an actual cache
hit, which is the property that was actually broken.
"""

from unittest.mock import patch

import numpy as np
import pytest

from kinematics_grading.analysis.embeddings import EmbeddingCache


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(120, 60, 200)).save(path)
    return path


def test_cache_key_is_os_independent_forward_slash_form(sample_image, tmp_path):
    key = EmbeddingCache._key(sample_image)
    assert "\\" not in key
    assert key == sample_image.resolve().as_posix()


def test_reload_is_a_real_cache_hit_not_a_silent_recompute(sample_image, tmp_path):
    cache_path = tmp_path / "cache.npz"

    cache1 = EmbeddingCache(cache_path)
    emb1 = cache1.get(sample_image)
    cache1.save()

    # Fresh instance, same file -- simulates a new process (or a later
    # pipeline stage) reusing the cache via --embeddings-cache.
    cache2 = EmbeddingCache(cache_path)
    with patch.object(EmbeddingCache, "_embed") as mock_embed:
        emb2 = cache2.get(sample_image)
        mock_embed.assert_not_called()  # this is the property that was broken

    assert np.allclose(emb1, emb2)


def test_reload_then_save_does_not_duplicate_entries(sample_image, tmp_path):
    cache_path = tmp_path / "cache.npz"

    cache1 = EmbeddingCache(cache_path)
    cache1.get(sample_image)
    cache1.save()

    cache2 = EmbeddingCache(cache_path)
    cache2.get(sample_image)  # should be a cache hit, not a fresh re-insert
    cache2.save()

    import zipfile

    with zipfile.ZipFile(cache_path) as zf:
        names = zf.namelist()
    assert len(names) == len(set(names)) == 1
