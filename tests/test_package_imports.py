def test_package_imports_version():
    import path_planner

    assert path_planner.__version__ == "0.1.0"


def test_readme_uses_shared_conda_environment_without_editable_install():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")

    assert "D:\\conda_envs\\lunar-explorer" in content
    assert "PYTHONPATH=src" in content
    assert "pip install -e" not in content


def test_dependency_metadata_caps_numpy_before_windows_matplotlib_crash_range():
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert '"numpy>=1.26,<2.3"' in content


def test_readme_describes_phase2_postprocess_scope_and_non_goals():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")

    assert "Phase 2" in content
    assert "corridor" in content
    assert "smoothed_path" in content
    assert "curvature_report" in content
    assert "does not implement GCS, IRIS, Ackermann trajectory optimization, Drake" in content
