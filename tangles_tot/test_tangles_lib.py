def test_tangle_library_imports():
    try:
        from ._tangles_lib import MetaData, FeatureSystem, Feature
    except ModuleNotFoundError:
        assert (
            False
        ), "Could not import necessary objects from the tangles library. Ensure that the tangle library is installed in your current environment as described in the README.md"
