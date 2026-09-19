import llm_eval


def test_package_import() -> None:
    assert llm_eval.__version__ == "0.1.0"
