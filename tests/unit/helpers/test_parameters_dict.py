"""
PD-* tests for app.helpers.miscellaneous.ParametersDict
"""

import pytest
from app.helpers.miscellaneous import ParametersDict, copy_mapping_data


@pytest.fixture
def defaults():
    return {"alpha": 1.0, "beta": "hello", "gamma": True}


@pytest.fixture
def pd(defaults):
    return ParametersDict({}, defaults)


# PD-01: missing key returns the default value
def test_missing_key_returns_default(pd, defaults):
    assert pd["alpha"] == defaults["alpha"]
    assert pd["beta"] == defaults["beta"]
    assert pd["gamma"] == defaults["gamma"]


# PD-02: explicitly set key wins over default
def test_set_key_overrides_default(defaults):
    pd = ParametersDict({"alpha": 99.9}, defaults)
    assert pd["alpha"] == 99.9


# PD-03: update() overwrites and both old + new keys are accessible
def test_update_merges_keys(pd, defaults):
    pd.update({"alpha": 42, "new_key": "new_val"})
    assert pd["alpha"] == 42
    assert pd["new_key"] == "new_val"
    # Keys already defaulted remain reachable via default
    assert pd["beta"] == defaults["beta"]


# PD-04: accessing a missing key via default does NOT mutate _default_parameters
def test_default_parameters_not_mutated(defaults):
    defaults_copy = dict(defaults)
    pd = ParametersDict({}, defaults)
    _ = pd["alpha"]  # trigger default path
    assert pd._default_parameters == defaults_copy


# PD-05: two independent ParametersDicts sharing the same default dict don't interfere
def test_independent_instances_share_defaults_safely(defaults):
    pd1 = ParametersDict({}, defaults)
    pd1["alpha"] = 100
    # pd2 should still see the original default, not pd1's override
    pd2_fresh = ParametersDict({}, defaults)
    assert pd2_fresh["alpha"] == defaults["alpha"]


# PD-06: .get(key, fallback) — UserDict.get() (via MutableMapping) calls __getitem__,
# so if a key is absent from self.data but present in _default_parameters the
# _default_parameters value is returned — the explicit fallback is ignored.
# Only keys absent from BOTH self.data and _default_parameters return the explicit fallback.
def test_get_with_explicit_fallback(defaults):
    pd = ParametersDict({}, defaults)
    # Key absent from data AND from defaults → explicit fallback returned
    assert pd.get("nonexistent", "fallback") == "fallback"
    # Key absent from data BUT present in defaults → __getitem__ returns the default value
    assert pd.get("alpha", "fallback") == defaults["alpha"]


def test_getitem_returns_default_not_get(defaults):
    pd = ParametersDict({}, defaults)
    # Direct key access triggers __getitem__ → returns ParametersDict default
    assert pd["alpha"] == defaults["alpha"]


# PD-07: bool value preserved through round-trip
def test_bool_value_round_trip(defaults):
    pd = ParametersDict({"gamma": False}, defaults)
    assert pd["gamma"] is False


# PD-08: len() and iteration work as expected
def test_len_and_iteration(defaults):
    pd = ParametersDict({"alpha": 5, "beta": "x"}, defaults)
    assert len(pd) == 2
    assert set(pd.keys()) == {"alpha", "beta"}


# PD-09: mapping copy helper must preserve ParametersDict contents
def test_copy_mapping_data_accepts_parameters_dict(defaults):
    pd = ParametersDict({"alpha": 5, "beta": "x"}, defaults)
    copied = copy_mapping_data(pd)
    assert copied == {"alpha": 5, "beta": "x"}
    assert isinstance(copied, dict)


# PD-10: non-mapping inputs fall back to an empty dict
def test_copy_mapping_data_rejects_non_mapping():
    assert copy_mapping_data(None) == {}
    assert copy_mapping_data(123) == {}
