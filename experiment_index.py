"""Executable index of the EpiLENS experiments included in this repository."""

EXPERIMENTS = {
    "PRQ-Net": "scripts/train_prq_net.py",
    "BCR-Net": "scripts/train_bcr_net.py",
    "CDEL": "scripts/evaluate_cdel.py",
    "component_ablations": "scripts/run_component_ablations.py",
    "quantile_sensitivity": "scripts/run_quantile_sensitivity.py",
    "strongest_baseline": "scripts/run_best_baseline.py",
    "fusion_permutation_bootstrap_boundary": "scripts/run_fusion_and_statistics.py",
    "cross_seizure": "scripts/run_cross_seizure_analysis.py",
    "leave_one_center_out": "scripts/run_leave_one_center_out.py",
    "leave_one_center_out_bootstrap": "scripts/run_loco_bootstrap.py",
}
