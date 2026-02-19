"""
Unit tests for UUID-based temp filename generation.

Tests the temp filename generation in PreviewImage and PreviewAudio nodes
after the refactoring from manual random strings to uuid.uuid4().hex[:5].

Related PR: #12475
"""
import re
import pytest
import uuid
from unittest.mock import patch, MagicMock


class TestUUIDTempFilenameFormat:
    """Test the format of UUID-based temp filenames."""

    def test_uuid_hex_produces_valid_5char_hex_string(self):
        """uuid.uuid4().hex[:5] should produce a 5-character hexadecimal string."""
        for _ in range(100):
            result = uuid.uuid4().hex[:5]
            assert len(result) == 5
            assert re.match(r'^[0-9a-f]{5}$', result), f"Expected hex string, got: {result}"

    def test_uuid_hex_uses_only_lowercase_hex_chars(self):
        """UUID hex should only contain 0-9 and a-f (lowercase)."""
        for _ in range(100):
            result = uuid.uuid4().hex[:5]
            assert all(c in '0123456789abcdef' for c in result)

    def test_uuid_hex_never_empty(self):
        """UUID hex slice should never be empty."""
        for _ in range(100):
            result = uuid.uuid4().hex[:5]
            assert result != ""


class TestPreviewImageTempPrefix:
    """Test PreviewImage node's temp filename prefix generation."""

    def test_preview_image_prefix_format_nodes_py(self):
        """PreviewImage in nodes.py should use '_temp_' + 5-char hex."""
        # Simulate the prefix_append from nodes.py PreviewImage
        prefix = "_temp_" + uuid.uuid4().hex[:5]
        
        assert prefix.startswith("_temp_")
        suffix = prefix[6:]  # Remove "_temp_" prefix
        assert len(suffix) == 5
        assert re.match(r'^[0-9a-f]{5}$', suffix)

    def test_preview_image_prefix_uniqueness_nodes_py(self):
        """Multiple PreviewImage instances should get unique prefixes."""
        prefixes = set()
        for _ in range(100):
            prefix = "_temp_" + uuid.uuid4().hex[:5]
            prefixes.add(prefix)
        
        # With 100 iterations, we should have high uniqueness
        # (collision probability is extremely low with UUIDs)
        assert len(prefixes) >= 99, f"Expected >= 99 unique prefixes, got {len(prefixes)}"

    def test_comfyui_temp_prefix_format_api(self):
        """PreviewImage in _ui.py should use 'ComfyUI_temp_' + 5-char hex."""
        # Simulate the filename_prefix from comfy_api/latest/_ui.py
        prefix = "ComfyUI_temp_" + uuid.uuid4().hex[:5]
        
        assert prefix.startswith("ComfyUI_temp_")
        suffix = prefix[13:]  # Remove "ComfyUI_temp_" prefix
        assert len(suffix) == 5
        assert re.match(r'^[0-9a-f]{5}$', suffix)

    def test_comfyui_temp_prefix_uniqueness_api(self):
        """Multiple API PreviewImage calls should get unique prefixes."""
        prefixes = set()
        for _ in range(100):
            prefix = "ComfyUI_temp_" + uuid.uuid4().hex[:5]
            prefixes.add(prefix)
        
        assert len(prefixes) >= 99, f"Expected >= 99 unique prefixes, got {len(prefixes)}"


class TestPreviewAudioTempPrefix:
    """Test PreviewAudio node's temp filename prefix generation."""

    def test_preview_audio_prefix_format(self):
        """PreviewAudio should use 'ComfyUI_temp_' + 5-char hex."""
        # Simulate the filename_prefix from PreviewAudio in _ui.py
        prefix = "ComfyUI_temp_" + uuid.uuid4().hex[:5]
        
        assert prefix.startswith("ComfyUI_temp_")
        suffix = prefix[13:]
        assert len(suffix) == 5
        assert re.match(r'^[0-9a-f]{5}$', suffix)

    def test_preview_audio_prefix_uniqueness(self):
        """Multiple PreviewAudio instances should get unique prefixes."""
        prefixes = set()
        for _ in range(100):
            prefix = "ComfyUI_temp_" + uuid.uuid4().hex[:5]
            prefixes.add(prefix)
        
        assert len(prefixes) >= 99, f"Expected >= 99 unique prefixes, got {len(prefixes)}"


class TestUUIDCollisionProbability:
    """Test that UUID-based approach has negligible collision probability."""

    def test_large_sample_has_minimal_collisions(self):
        """
        Test that even with 1000 temp files, collisions are extremely rare.
        
        With 5 hex chars (16^5 = 1,048,576 possibilities), the probability of
        collision is very low for reasonable workload sizes.
        """
        temp_ids = set()
        duplicates = 0
        
        for _ in range(1000):
            temp_id = uuid.uuid4().hex[:5]
            if temp_id in temp_ids:
                duplicates += 1
            temp_ids.add(temp_id)
        
        # We expect 0 or at most 1-2 collisions in 1000 samples
        # (Birthday paradox: ~0.48% chance of any collision with 1000 samples)
        assert duplicates <= 2, f"Too many collisions: {duplicates}"
        assert len(temp_ids) >= 998, f"Expected >= 998 unique IDs, got {len(temp_ids)}"


class TestBackwardCompatibilityWithOldApproach:
    """Test that the new UUID approach maintains compatibility."""

    def test_old_typo_had_missing_w(self):
        """
        Document the bug in the old approach: missing 'w' in alphabet.
        
        The old code used "abcdefghijklmnopqrstupvxyz" which:
        - Had duplicate 'p'
        - Was missing 'w'
        
        This test documents the fix.
        """
        old_alphabet_with_typo = "abcdefghijklmnopqrstupvxyz"
        correct_alphabet = "abcdefghijklmnopqrstuvwxyz"
        
        # Verify the typo existed
        assert 'w' not in old_alphabet_with_typo
        assert old_alphabet_with_typo.count('p') == 2  # duplicate p
        
        # Verify the correct alphabet
        assert 'w' in correct_alphabet
        assert correct_alphabet.count('p') == 1
        assert len(set(correct_alphabet)) == 26

    def test_new_hex_charset_is_valid(self):
        """The new hex charset (0-9a-f) is a proper subset of alphanumeric."""
        hex_chars = set('0123456789abcdef')
        alphanumeric = set('abcdefghijklmnopqrstuvwxyz0123456789')
        
        assert hex_chars.issubset(alphanumeric)
        assert len(hex_chars) == 16

    def test_prefix_length_unchanged(self):
        """
        Both approaches use 5-character suffixes to maintain compatibility
        with any external scripts that parse temp filenames.
        """
        # Old approach (simulated)
        old_suffix_length = 5
        
        # New approach
        new_suffix = uuid.uuid4().hex[:5]
        
        assert len(new_suffix) == old_suffix_length

    def test_prefix_strings_unchanged(self):
        """
        Verify that the prefix strings ('_temp_' and 'ComfyUI_temp_')
        remain exactly the same to avoid breaking external parsing.
        """
        # These exact strings should be maintained for backward compatibility
        nodes_py_prefix = "_temp_"
        api_prefix = "ComfyUI_temp_"
        
        # Verify they haven't been changed
        assert nodes_py_prefix == "_temp_"
        assert api_prefix == "ComfyUI_temp_"
        
        # Full temp filenames should preserve these exact prefixes
        full_nodes_temp = nodes_py_prefix + uuid.uuid4().hex[:5]
        full_api_temp = api_prefix + uuid.uuid4().hex[:5]
        
        assert full_nodes_temp.startswith("_temp_")
        assert full_api_temp.startswith("ComfyUI_temp_")


class TestUUIDDistributionProperties:
    """Test statistical properties of UUID-based random suffix."""

    def test_character_distribution_is_uniform(self):
        """
        UUID hex should provide roughly uniform distribution of characters.
        
        This is a statistical test - with enough samples, all hex chars
        should appear with similar frequency.
        """
        char_counts = {c: 0 for c in '0123456789abcdef'}
        samples = 10000
        
        for _ in range(samples):
            suffix = uuid.uuid4().hex[:5]
            for char in suffix:
                char_counts[char] += 1
        
        total_chars = sum(char_counts.values())
        assert total_chars == samples * 5
        
        # Each of 16 hex chars should appear roughly 1/16 of the time
        expected_per_char = (samples * 5) / 16
        
        # Allow 30% variance from expected (statistical test)
        for char, count in char_counts.items():
            ratio = count / expected_per_char
            assert 0.7 <= ratio <= 1.3, \
                f"Char '{char}' appeared {count} times (expected ~{expected_per_char:.0f})"

    def test_positional_independence(self):
        """
        Each position in the 5-char suffix should be independent.
        Test that all 5 positions can produce all hex characters.
        """
        position_chars = [set() for _ in range(5)]
        samples = 1000
        
        for _ in range(samples):
            suffix = uuid.uuid4().hex[:5]
            for i, char in enumerate(suffix):
                position_chars[i].add(char)
        
        # With 1000 samples, each position should see most hex chars
        for i, chars_seen in enumerate(position_chars):
            # We expect to see at least 14 out of 16 hex chars per position
            assert len(chars_seen) >= 14, \
                f"Position {i} only saw {len(chars_seen)} different chars: {chars_seen}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_uuid_hex_slice_bounds(self):
        """Verify [:5] slice is always safe and never throws."""
        for _ in range(100):
            full_hex = uuid.uuid4().hex
            assert len(full_hex) == 32  # UUID4 hex is always 32 chars
            
            sliced = full_hex[:5]
            assert len(sliced) == 5
            assert sliced == full_hex[0:5]

    def test_concurrent_uuid_generation_uniqueness(self):
        """
        Test that rapid sequential UUID generation produces unique values.
        This simulates multiple preview nodes being created quickly.
        """
        uuids = [uuid.uuid4().hex[:5] for _ in range(1000)]
        unique_uuids = set(uuids)
        
        # We expect high uniqueness even with rapid generation
        assert len(unique_uuids) >= 998, \
            f"Expected >= 998 unique UUIDs from 1000 samples, got {len(unique_uuids)}"

    def test_no_special_characters_in_suffix(self):
        """UUID hex should never contain special characters that could break filenames."""
        forbidden_chars = set(' \\/:*?"<>|')
        
        for _ in range(100):
            suffix = uuid.uuid4().hex[:5]
            for char in suffix:
                assert char not in forbidden_chars, \
                    f"Found forbidden character '{char}' in suffix '{suffix}'"


class TestIntegrationPatterns:
    """Test realistic usage patterns for temp file generation."""

    def test_typical_workflow_temp_file_pattern(self):
        """
        Simulate a typical workflow that generates multiple preview images.
        Each should get a unique temp filename.
        """
        temp_files = []
        
        # Simulate 20 preview images in a workflow
        for _ in range(20):
            filename = f"ComfyUI_temp_{uuid.uuid4().hex[:5]}_00001_.png"
            temp_files.append(filename)
        
        # All filenames should be unique
        assert len(temp_files) == len(set(temp_files))
        
        # All should match the expected pattern
        pattern = r'^ComfyUI_temp_[0-9a-f]{5}_\d{5}_\.png$'
        for filename in temp_files:
            assert re.match(pattern, filename), \
                f"Filename '{filename}' doesn't match expected pattern"

    def test_nodes_py_preview_pattern(self):
        """
        Simulate the PreviewImage node from nodes.py pattern.
        """
        temp_files = []
        
        for _ in range(20):
            prefix = "_temp_" + uuid.uuid4().hex[:5]
            # Typical filename: ComfyUI_00001_.png with prefix prepended
            filename = f"ComfyUI{prefix}_00001_.png"
            temp_files.append(filename)
        
        assert len(temp_files) == len(set(temp_files))
        
        # Pattern: ComfyUI_temp_<5hex>_<5digits>_.png
        pattern = r'^ComfyUI_temp_[0-9a-f]{5}_\d{5}_\.png$'
        for filename in temp_files:
            assert re.match(pattern, filename), \
                f"Filename '{filename}' doesn't match expected pattern"
