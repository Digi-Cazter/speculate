import re
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

class ExpectationMixin:
    """
    Mixin providing expectation builder methods.
    Assumes the host class defines:
      self._expect_kind
      self._expect_value
      self._expect_regex
      self._expect_model
      self._expect_model_values
      self._expect_coerce_json
    """

    def expect_exact(self, expected: str):
        self._expect_kind = "exact"
        self._expect_value = expected
        return self

    def expect_contains(self, substring: str):
        self._expect_kind = "contains"
        self._expect_value = substring
        return self

    def expect_not_equal(self, unexpected: str):
        self._expect_kind = "not_equal"
        self._expect_value = unexpected
        return self

    def expect_not_contains(self, substring: str):
        self._expect_kind = "not_contains"
        self._expect_value = substring
        return self

    def expect_regex(self, pattern: str):
        self._expect_kind = "regex"
        self._expect_regex = re.compile(pattern, re.DOTALL)
        return self

    def expect_schema(self, model: Type[BaseModel], coerce_json: bool = True):
        """
        Expect output to be valid JSON (optionally coerced) that matches the Pydantic model.
        Does NOT assert any specific field values.
        On failure, any missing required fields will be listed in the mismatch details.
        """
        self._expect_kind = "schema"
        self._expect_model = model
        self._expect_model_values = {}
        self._expect_coerce_json = coerce_json
        return self

    def expect_schema_values(self, model: Type[BaseModel], coerce_json: bool = True, **expected_fields: Any):
        """
        Expect output to match the model AND specified field values.
        Example:
            .expect_schema_values(MyModel, field1="x", count=3)
        """
        self._expect_kind = "schema"
        self._expect_model = model
        self._expect_model_values = expected_fields
        self._expect_coerce_json = coerce_json
        return self
