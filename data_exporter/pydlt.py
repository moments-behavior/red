"""
Convert perspective camera parameters (intrinsics + extrinsics) to DLT projection matrix.
For a perspective camera: P = K[R|t] where P is 3x4
"""

import numpy as np
import cv2
import os
import argparse
from pathlib import Path


def compute_dlt_matrix(intrinsic_matrix, rotation_matrix, translation_vector):
    """
    Compute DLT projection matrix from camera parameters.
    
    Args:
        intrinsic_matrix: 3x3 camera intrinsic matrix K
        rotation_matrix: 3x3 rotation matrix R
        translation_vector: 3x1 or (3,) translation vector t
        
    Returns:
        projection_matrix: 3x4 DLT projection matrix P = K[R|t]
    """
    # Ensure translation is column vector
    if translation_vector.ndim == 1:
        translation_vector = translation_vector.reshape(3, 1)
    
    # Construct extrinsic matrix [R|t]
    extrinsic_matrix = np.hstack([rotation_matrix, translation_vector])  # 3x4
    
    # Compute projection matrix P = K * [R|t]
    projection_matrix = intrinsic_matrix @ extrinsic_matrix  # 3x4
    print("Computed projection matrix P:")
    print(projection_matrix)
    
    return projection_matrix


def load_camera_params_from_yaml(yaml_path):
    """
    Load camera parameters from OpenCV YAML/XML calibration file.
    
    Returns:
        intrinsic_matrix: 3x3
        rotation_matrix: 3x3
        translation_vector: 3x1
        distortion_coefficients: 1x5 (optional)
    """
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    
    intrinsic_matrix = fs.getNode('intrinsicMatrix').mat()
    rotation_matrix = fs.getNode('R').mat()
    translation_vector = fs.getNode('T').mat()
    distortion_coefficients = fs.getNode('distortionCoefficients').mat()
    
    fs.release()
    
    return intrinsic_matrix, rotation_matrix, translation_vector, distortion_coefficients


def save_dlt_matrix_to_yaml(output_path, projection_matrix, 
                            intrinsic_matrix=None, 
                            rotation_matrix=None, 
                            translation_vector=None,
                            distortion_coefficients=None,
                            projection_only=False):
    """
    Save DLT projection matrix to YAML file in OpenCV format.
    Optionally save original parameters for reference.
    
    Args:
        output_path: Path to output YAML file
        projection_matrix: 3x4 projection matrix
        intrinsic_matrix: 3x3 intrinsic matrix (optional)
        rotation_matrix: 3x3 rotation matrix (optional)
        translation_vector: 3x1 translation vector (optional)
        distortion_coefficients: distortion coefficients (optional)
        projection_only: If True, only write projection matrix
    """
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    
    # Write projection matrix
    fs.write('projectionMatrix', projection_matrix)
    
    # Optionally write original parameters for reference
    if not projection_only:
        if intrinsic_matrix is not None:
            fs.write('intrinsicMatrix', intrinsic_matrix)
        if rotation_matrix is not None:
            fs.write('R', rotation_matrix)
        if translation_vector is not None:
            fs.write('T', translation_vector)
        if distortion_coefficients is not None:
            fs.write('distortionCoefficients', distortion_coefficients)
    
    fs.release()
    
    if projection_only:
        print(f"Saved DLT projection matrix only to: {output_path}")
    else:
        print(f"Saved DLT projection matrix with original parameters to: {output_path}")


def save_dlt_parameters_to_yaml(output_path, projection_matrix, projection_only=False, 
                                intrinsic_matrix=None, rotation_matrix=None, 
                                translation_vector=None, distortion_coefficients=None):
    """
    Save 11 DLT parameters to YAML file.
    Normalizes P[2,3] to 1.0 and saves the first 11 elements.
    """
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    
    # Normalize projection matrix so P[2,3] = 1
    P_normalized = projection_matrix / projection_matrix[2, 3]
    
    # Extract 11 parameters (all except P[2,3] which is now 1)
    dlt_params = P_normalized.flatten()[:11].reshape(11, 1)
    
    # Write 11 DLT parameters
    fs.write('dlt_parameters', dlt_params)
    
    # Also write full projection matrix for reference
    fs.write('projectionMatrix', projection_matrix)
    
    if not projection_only:
        if intrinsic_matrix is not None:
            fs.write('intrinsicMatrix', intrinsic_matrix)
        if rotation_matrix is not None:
            fs.write('R', rotation_matrix)
        if translation_vector is not None:
            fs.write('T', translation_vector)
        if distortion_coefficients is not None:
            fs.write('distortionCoefficients', distortion_coefficients)
    
    fs.release()
    print(f"Saved 11 DLT parameters to: {output_path}")


def load_dlt_parameters_from_csv(csv_path):
    """
    Load 11x1 DLT parameters from a CSV file and reconstruct the projection matrix.

    Args:
        csv_path: Path to the CSV file containing 11 DLT parameters.

    Returns:
        projection_matrix: 3x4 projection matrix reconstructed from the DLT parameters.
    """
    dlt_params = np.loadtxt(csv_path, delimiter=',').reshape(11, 1)

    # Append a 1 at the end of dlt_params
    dlt_params = np.append(dlt_params, 1)
    
    # Reconstruct the projection matrix
    projection_matrix = np.zeros((3, 4))
    projection_matrix = dlt_params.reshape(3, 4)
    
    print(f"Loaded DLT parameters from: {csv_path}")
    print(f"Reconstructed projection matrix:\n{projection_matrix}")
    
    return projection_matrix


def convert_csv_directory_to_yaml(input_dir, output_dir=None, pattern="*.csv"):
    """
    Convert all CSV files with 11x1 DLT parameters in a directory to YAML files.

    Args:
        input_dir: Directory containing CSV files with DLT parameters.
        output_dir: Output directory for YAML files (if None, uses input_dir).
        pattern: File pattern to match CSV files (default: *.csv).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob(pattern))
    
    if not csv_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files with DLT parameters")
    
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        output_file = output_path / f"{csv_file.stem}.yaml"
        try:
            # Load DLT parameters and reconstruct projection matrix
            projection_matrix = load_dlt_parameters_from_csv(str(csv_file))
            
            # Save projection matrix to YAML
            save_dlt_matrix_to_yaml(output_file, projection_matrix, projection_only=True)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


def convert_calibration_file(input_yaml, output_yaml=None, projection_only=False):
    """
    Convert a perspective camera calibration file to DLT format.
    
    Args:
        input_yaml: Path to input calibration YAML
        output_yaml: Path to output YAML (if None, adds '_dlt' suffix)
        projection_only: If True, only save projection matrix
    """
    # Load camera parameters
    K, R, t, dist = load_camera_params_from_yaml(input_yaml)
    
    print(f"Loaded camera parameters from: {input_yaml}")
    print(f"Intrinsic matrix K:\n{K}")
    print(f"Rotation matrix R:\n{R}")
    print(f"Translation vector t:\n{t.T}")
    
    # Compute DLT projection matrix
    P = compute_dlt_matrix(K, R, t)
    
    print(f"\nComputed DLT projection matrix P:\n{P}")
    
    # Determine output path
    if output_yaml is None:
        input_path = Path(input_yaml)
        output_yaml = input_path.parent / f"{input_path.stem}_dlt{input_path.suffix}"
    
    # Save to file
    if projection_only:
        save_dlt_matrix_to_yaml(output_yaml, P, projection_only=True)
    else:
        save_dlt_matrix_to_yaml(output_yaml, P, K, R, t, dist, projection_only=False)
    
    return P


def convert_directory(input_dir, output_dir=None, pattern="*.yaml", projection_only=False):
    """
    Convert all calibration files in a directory.
    
    Args:
        input_dir: Directory containing calibration YAML files
        output_dir: Output directory (if None, uses input_dir)
        pattern: File pattern to match
        projection_only: If True, only save projection matrices
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    yaml_files = list(input_path.glob(pattern))
    
    if not yaml_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(yaml_files)} calibration files")
    
    for yaml_file in yaml_files:
        print(f"\n{'='*60}")
        output_file = output_path / yaml_file.name
        try:
            convert_calibration_file(str(yaml_file), str(output_file), projection_only)
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")


def verify_projection(P, K, R, t, test_point_3d=None):
    """
    Verify that P = K[R|t] by projecting a test point.
    
    Args:
        P: 3x4 projection matrix
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        test_point_3d: 3D point to project (if None, uses [0, 0, 1])
    """
    if test_point_3d is None:
        test_point_3d = np.array([0.0, 0.0, 1.0, 1.0])  # Homogeneous coordinates
    else:
        test_point_3d = np.append(test_point_3d, 1.0)  # Add homogeneous coordinate
    
    # Project using DLT matrix
    p_dlt = P @ test_point_3d
    p_dlt = p_dlt[:2] / p_dlt[2]  # Normalize
    
    # Project using K[R|t]
    extrinsic = np.hstack([R, t.reshape(3, 1)])
    p_separate = K @ extrinsic @ test_point_3d
    p_separate = p_separate[:2] / p_separate[2]  # Normalize
    
    print("\nVerification:")
    print(f"Projected point using DLT: {p_dlt}")
    print(f"Projected point using K[R|t]: {p_separate}")
    print(f"Difference: {np.linalg.norm(p_dlt - p_separate):.2e}")
    
    return np.allclose(p_dlt, p_separate, atol=1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert perspective camera calibration to DLT projection matrix"
    )
    parser.add_argument(
        "input",
        help="Input calibration YAML file, CSV file, or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output YAML file or directory (optional)"
    )
    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Process entire directory"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.yaml",
        help="File pattern for directory mode (default: *.yaml)"
    )
    parser.add_argument(
        "--csv-to-yaml",
        action="store_true",
        help="Convert CSV files with DLT parameters to YAML"
    )
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Only write projection matrix (don't keep original parameters)"
    )
    parser.add_argument(
        "-v", "--verify",
        action="store_true",
        help="Verify projection matrix computation"
    )
    
    args = parser.parse_args()
    
    if args.csv_to_yaml:
        print("Converting CSV files with DLT parameters to YAML format")
        convert_csv_directory_to_yaml(
            args.input,
            args.output,
            pattern="*.csv"
        )
    elif args.directory:
        convert_directory(
            args.input,
            args.output,
            args.pattern,
            projection_only=args.projection_only
        )
    else:
        P = convert_calibration_file(
            args.input,
            args.output,
            projection_only=args.projection_only
        )
        
        if args.verify:
            K, R, t, _ = load_camera_params_from_yaml(args.input)
            verify_projection(P, K, R, t)


