"""Schema-level unit tests for ListAssetsQuery (no DB required)."""
import uuid

import pytest
from pydantic import ValidationError

from app.assets.api.schemas_in import ListAssetsQuery


class TestJobIdsValidator:
    def test_csv_string_parses_and_canonicalizes(self):
        a = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
        b = "11111111-2222-3333-4444-555555555555"
        q = ListAssetsQuery.model_validate({"job_ids": f"{a},{b}"})
        # Canonicalized to lowercase
        assert q.job_ids == [a.lower(), b]

    def test_repeated_query_params_as_list(self):
        a = "11111111-1111-1111-1111-111111111111"
        b = "22222222-2222-2222-2222-222222222222"
        q = ListAssetsQuery.model_validate({"job_ids": [a, b]})
        assert q.job_ids == [a, b]

    def test_dedup_preserves_first_seen_order(self):
        a = "11111111-1111-1111-1111-111111111111"
        b = "22222222-2222-2222-2222-222222222222"
        q = ListAssetsQuery.model_validate({"job_ids": [a, b, a]})
        assert q.job_ids == [a, b]

    def test_default_empty(self):
        q = ListAssetsQuery.model_validate({})
        assert q.job_ids == []

    def test_invalid_uuid_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ListAssetsQuery.model_validate({"job_ids": "not-a-uuid"})
        assert "must be UUIDs" in str(exc.value)

    def test_non_string_list_item_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ListAssetsQuery.model_validate(
                {"job_ids": ["11111111-1111-1111-1111-111111111111", 42]}
            )
        assert "must be strings" in str(exc.value)

    def test_non_string_non_list_value_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ListAssetsQuery.model_validate({"job_ids": {"bad": "shape"}})
        assert "must be a string or list of strings" in str(exc.value)

    def test_max_length_enforced(self):
        too_many = [str(uuid.uuid4()) for _ in range(501)]
        with pytest.raises(ValidationError) as exc:
            ListAssetsQuery.model_validate({"job_ids": too_many})
        assert exc.value.errors()[0]["type"] == "too_long"

    def test_max_length_boundary_accepted(self):
        at_cap = [str(uuid.uuid4()) for _ in range(500)]
        q = ListAssetsQuery.model_validate({"job_ids": at_cap})
        assert len(q.job_ids) == 500
