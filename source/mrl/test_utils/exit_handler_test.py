import pytest
from mrl.test_utils.error_handler import describe_exception


@pytest.mark.quick
def test_describe_exception_expands_key_error():
    assert describe_exception(KeyError('oracle')) == "KeyError: missing key 'oracle'."


@pytest.mark.quick
def test_describe_exception_includes_cause_chain():
    try:
        try:
            raise KeyError('policy')
        except KeyError as error:
            raise TypeError("Invalid configuration.") from error
    except TypeError as error:
        assert describe_exception(error) == (
            "TypeError: Invalid configuration.\n"
            "Caused by: KeyError: missing key 'policy'."
        )
