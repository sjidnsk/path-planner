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


def test_readme_describes_phase4_trackable_path_scope_and_non_goals():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")

    assert "Phase 4" in content
    assert "trackable_path" in content
    assert "TrackingSafetyReport" in content
    assert "speed_profile" in content
    assert "Phase 5" in content
    assert "tracking_simulation_report" in content
    assert "Simulated Tracking Path" in content
    assert "max_cross_track_error_m" in content
    assert "Phase 6" in content
    assert "trajectory_optimization_report" in content
    assert "Optimized Path" in content
    assert "fixed-corridor continuous trajectory optimization" in content
    assert "Phase 7" in content
    assert "resampled_optimized_path" in content
    assert "Execution-Aware Optimization Summary" in content
    assert "tracking_error_proxy" in content
    assert "Phase 3" in content
    assert "platform-aware A*" in content
    assert "PlanningGrid" in content
    assert "PlanningConstraints" in content
    assert "inflated_passable_mask" in content
    assert "Phase 2" in content
    assert "corridor" in content
    assert "smoothed_path" in content
    assert "curvature_report" in content
    assert "Phase 8" in content
    assert "Drake IRIS/GCS framework prototype" in content
    assert "original 2D workspace `Iris`" in content
    assert "Phase 8.1" in content
    assert "grid_box" in content
    assert "blocked_cell_box" in content
    assert "merged_blocked_rectangle" in content
    assert "postprocess_corridor_safe_component_box" in content
    assert "Phase 8.2" in content
    assert "Phase 8.3" in content
    assert "--drake-iris-regions" in content
    assert "workspace_iris" in content
    assert "iris_region_report" in content
    assert "region_graph_report" in content
    assert "quality_metrics" in content
    assert "start_goal_connected" in content
    assert "IRIS / Region Graph Summary" in content
    assert "gcs_trajectory_report" in content
    assert "pydrake_direction_cone_program" in content
    assert "rho_source_counts" in content
    assert "candidate_decision" in content
    assert "quality_gate" in content
    assert "path_planner.drake_backend.gcs_cli_batch" in content
    assert "gcs_direction_cone_cli_scenario_batch/v1" in content
    assert "gcs_motion_feasibility_cli_batch/v1" in content
    assert "--batch-kind motion-feasibility" in content
    assert "not an Ackermann trajectory optimizer" in content
    assert "does not yet implement full GCS graph search" in content
    assert "Bezier/B-spline GCS trajectory optimization" in content
    assert "candidate future reports" not in content


def test_phase8_drake_documents_exist_and_define_framework_first_scope():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    knowledge = root / "docs" / "drake" / "pydrake-knowledge-structure.md"
    spec = root / "docs" / "superpowers" / "specs" / "2026-05-30-path-planner-phase8-drake-iris-gcs-design.md"
    plan = root / "docs" / "superpowers" / "plans" / "2026-05-30-path-planner-phase8-drake-iris-gcs-framework.md"

    for path in (knowledge, spec, plan):
        assert path.exists(), f"missing {path}"

    knowledge_text = knowledge.read_text(encoding="utf-8")
    spec_text = spec.read_text(encoding="utf-8")
    plan_text = plan.read_text(encoding="utf-8")

    assert "https://drake.mit.edu/pydrake/index.html" in knowledge_text
    assert "GraphOfConvexSets" in knowledge_text
    assert "GcsTrajectoryOptimization" in knowledge_text
    assert "IrisNp2" in knowledge_text
    assert "IrisZo" in knowledge_text
    assert "非凸" in knowledge_text
    assert "Ackermann" in knowledge_text
    assert "experimental" in knowledge_text
    assert "original 2D workspace `Iris`" in spec_text
    assert "convex obstacle primitives" in spec_text
    assert "motion_feasibility_report" in spec_text
    assert "ObstaclePrimitive" in spec_text
    assert "RegionGraphReport" in spec_text
    assert "SampledTrajectory" in spec_text
    assert "backend_unavailable" in spec_text
    assert "--drake-iris-regions" in spec_text
    assert "iris_region_report" in spec_text
    assert "Phase 8.3" in spec_text
    assert "sampled_connectivity" in spec_text
    assert "quality_metrics" in spec_text
    assert "start_goal_connected" in spec_text
    assert "IRIS / Region Graph Summary" in spec_text
    assert "fallback_used" in spec_text
    assert "Drake unavailable" in spec_text
    assert "current fixed-corridor optimizer" in spec_text
    assert "raw A* geometric_path" in spec_text
    assert "trajectory_kind" in spec_text
    assert "geometric_path" in spec_text
    assert "ObstaclePrimitive" in plan_text
    assert "workspace_iris" in plan_text
    assert "Framework First" in plan_text
    assert "pydrake" in plan_text
