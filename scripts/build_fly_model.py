#!/usr/bin/env python3
"""Build the fruitfly MuJoCo model with 50 keypoint sites.

The fruitfly v2 model from janelia-anibody/fruitfly uses OBJ mesh files
which require the MuJoCo Python package's built-in decoder (the C framework
doesn't include the OBJ plugin). This script:

1. Clones the fruitfly repo (if not already present)
2. Adds 50 keypoint sites matching RED's Fly50 skeleton
3. Compiles and saves as .mjb binary (all meshes baked in)

Usage:
    pip install mujoco
    python3 scripts/build_fly_model.py

Output:
    models/fruitfly/fruitfly_fly50.mjb

Source: https://github.com/janelia-anibody/fruitfly
Sites: https://github.com/TuragaLab/fly-body-tuning (add_keypoint_sites)
"""

import os
import subprocess
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    red_root = os.path.dirname(script_dir)
    model_dir = os.path.join(red_root, "models", "fruitfly")
    output_path = os.path.join(model_dir, "fruitfly_fly50.mjb")

    # Clone fruitfly repo if needed
    fruitfly_dir = os.path.join(red_root, "lib", "fruitfly")
    xml_path = os.path.join(fruitfly_dir, "fruitfly_v2", "assets", "fruitfly.xml")

    if not os.path.exists(xml_path):
        print(f"Cloning janelia-anibody/fruitfly to {fruitfly_dir}...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/janelia-anibody/fruitfly.git",
            fruitfly_dir
        ], check=True)

    if not os.path.exists(xml_path):
        print(f"ERROR: {xml_path} not found after clone")
        sys.exit(1)

    # Import mujoco
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco Python package required. Install with: pip install mujoco")
        sys.exit(1)

    print(f"Loading {xml_path}...")
    spec = mujoco.MjSpec.from_file(xml_path)

    # Fly50 keypoint site definitions
    # From: TuragaLab/fly-body-tuning projectlib/_inverse_kinematics.py
    fly50_sites = [
        # Head/body/wing keypoints
        ("Antenna_Base", "head",              [0.0, 0.038, 0.012]),
        ("EyeL",         "head",              [-0.0245, 0.0135, 0.0285]),
        ("EyeR",         "head",              [0.0245, 0.0135, 0.0285]),
        ("Scutellum",    "thorax",            [-0.049, 0.0, 0.04]),
        ("Abd_A4",       "abdomen_3",         [0.0, 0.0335, 0.021]),
        ("Abd_tip",      "abdomen_7",         [0.0, 0.0395, -0.001]),
        ("WingL_base",   "thorax",            [-0.0095, 0.045, 0.0175]),
        ("WingL_V12",    "wing_left",         [-0.0072, -0.2125, 0.0075]),
        ("WingL_V13",    "wing_left",         [0.0221, -0.2562, -0.0253]),
        # Left legs
        ("T1L_ThxCx",    "coxa_T1_left",      [0, 0, 0]),
        ("T1L_Tro",      "femur_T1_left",     [0, 0, 0]),
        ("T1L_FeTi",     "tibia_T1_left",     [0, 0, 0]),
        ("T1L_TiTa",     "tarsus1_T1_left",   [0, 0, 0]),
        ("T1L_TaT1",     "tarsus2_T1_left",   [0, 0, 0]),
        ("T1L_TaT3",     "tarsus4_T1_left",   [0, 0, 0]),
        ("T1L_TaTip",    "tarsal_claw_T1_left", [0, 0.0105, 0.0006]),
        ("T2L_Tro",      "femur_T2_left",     [0, 0, 0]),
        ("T2L_FeTi",     "tibia_T2_left",     [0, 0, 0]),
        ("T2L_TiTa",     "tarsus1_T2_left",   [0, 0, 0]),
        ("T2L_TaT1",     "tarsus2_T2_left",   [0, 0, 0]),
        ("T2L_TaT3",     "tarsus4_T2_left",   [0, 0, 0]),
        ("T2L_TaTip",    "tarsal_claw_T2_left", [0, 0.0122, 0.0006]),
        ("T3L_Tro",      "femur_T3_left",     [0, 0, 0]),
        ("T3L_FeTi",     "tibia_T3_left",     [0, 0, 0]),
        ("T3L_TiTa",     "tarsus1_T3_left",   [0, 0, 0]),
        ("T3L_TaT1",     "tarsus2_T3_left",   [0, 0, 0]),
        ("T3L_TaT3",     "tarsus4_T3_left",   [0, 0, 0]),
        ("T3L_TaTip",    "tarsal_claw_T3_left", [0, 0.0111, 0.0008]),
        # Right wing
        ("WingR_base",   "thorax",            [-0.0095, -0.045, 0.0175]),
        ("WingR_V12",    "wing_right",        [0.0072, 0.2125, -0.0075]),
        ("WingR_V13",    "wing_right",        [-0.0221, 0.2562, 0.0253]),
        # Right legs
        ("T1R_ThxCx",    "coxa_T1_right",     [0, 0, 0]),
        ("T1R_Tro",      "femur_T1_right",    [0, 0, 0]),
        ("T1R_FeTi",     "tibia_T1_right",    [0, 0, 0]),
        ("T1R_TiTa",     "tarsus1_T1_right",  [0, 0, 0]),
        ("T1R_TaT1",     "tarsus2_T1_right",  [0, 0, 0]),
        ("T1R_TaT3",     "tarsus4_T1_right",  [0, 0, 0]),
        ("T1R_TaTip",    "tarsal_claw_T1_right", [0, -0.0101, -0.0006]),
        ("T2R_Tro",      "femur_T2_right",    [0, 0, 0]),
        ("T2R_FeTi",     "tibia_T2_right",    [0, 0, 0]),
        ("T2R_TiTa",     "tarsus1_T2_right",  [0, 0, 0]),
        ("T2R_TaT1",     "tarsus2_T2_right",  [0, 0, 0]),
        ("T2R_TaT3",     "tarsus4_T2_right",  [0, 0, 0]),
        ("T2R_TaTip",    "tarsal_claw_T2_right", [0, -0.0118, -0.0006]),
        ("T3R_Tro",      "femur_T3_right",    [0, 0, 0]),
        ("T3R_FeTi",     "tibia_T3_right",    [0, 0, 0]),
        ("T3R_TiTa",     "tarsus1_T3_right",  [0, 0, 0]),
        ("T3R_TaT1",     "tarsus2_T3_right",  [0, 0, 0]),
        ("T3R_TaT3",     "tarsus4_T3_right",  [0, 0, 0]),
        ("T3R_TaTip",    "tarsal_claw_T3_right", [0, -0.0109, -0.0008]),
    ]

    # Add sites
    added = 0
    for name, body_name, pos in fly50_sites:
        body = spec.body(body_name)
        if body is None:
            print(f"  WARNING: body '{body_name}' not found for site '{name}'")
            continue
        site = body.add_site()
        site.name = name
        site.pos = pos
        added += 1

    print(f"Added {added}/50 keypoint sites")

    # Compile
    model = spec.compile()
    print(f"Compiled: {model.nbody} bodies, {model.njnt} joints, {model.nsite} sites, {model.nmesh} meshes")

    # Save
    os.makedirs(model_dir, exist_ok=True)
    mujoco.mj_saveModel(model, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
