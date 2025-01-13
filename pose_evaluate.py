import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from lightglue import SuperPoint, LightGlue, SIFT
from lightglue.utils import numpy_image_to_torch, rbd
import cv2

def compute_auc(errors, thresholds):
    """
    Compute the AUC (Area Under Curve) for given errors and thresholds.
    
    Args:
        errors: List of pose errors
        thresholds: List of thresholds to evaluate at
        
    Returns:
        aucs: Dictionary of AUC values for each threshold
    """
    errors = np.array(errors)
    aucs = {}
    
    for thresh in thresholds:
        # Calculate accuracy at this threshold
        accuracy = np.mean((errors <= thresh).astype(float)) * 100
        aucs[f'auc_{thresh}'] = accuracy
        
    return aucs

def load_pose_data(txt_path):
    """Load camera poses and timestamps from RealEstate10K text file."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first line (header) and process the rest
    data_lines = [line.strip().split() for line in lines[1:]]
    data = np.array(data_lines)
    
    timestamps = data[:, 0].astype(float)
    intrinsics = data[:, 1:7].astype(float)  # fx, fy, cx, cy, k1, k2
    poses = data[:, 7:].astype(float).reshape(-1, 3, 4)  # R|t
    return timestamps, intrinsics, poses

def compute_pose_error(T_pred, T_gt):
    """Compute pose error metrics between predicted and ground truth poses."""
    # Extract rotation and translation
    R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    
    # Rotation error (degrees)
    R_error = np.arccos(np.clip((np.trace(R_pred @ R_gt.T) - 1) / 2, -1, 1))
    R_error = np.rad2deg(R_error)
    
    # Translation error (degrees)
    t_error = np.arccos(np.clip(np.dot(t_pred, t_gt) / 
                               (np.linalg.norm(t_pred) * np.linalg.norm(t_gt)), -1, 1))
    t_error = np.rad2deg(t_error)
    
    return R_error, t_error


def estimate_pose(kpts0, kpts1, K1, K2, use_loransac=False, thresh=0.5):        # default thresh was 1e-4 
    """Estimate relative pose from matches using 5-point algorithm."""
    # Normalize keypoints
    kpts0_norm = cv2.undistortPoints(kpts0.reshape(-1,1,2), K1, None)
    kpts1_norm = cv2.undistortPoints(kpts1.reshape(-1,1,2), K1, None)

    # return None if kpts0 and kpts1 are the exact same
    if np.allclose(kpts0_norm, kpts1_norm):
        return None, None, None
    
    f_mean = (K1[0, 0] + K1[1, 1]) / 2
    norm_thresh = thresh / f_mean

    if use_loransac:
        # Use LO-RANSAC
        E, mask = cv2.findEssentialMat(
            kpts0_norm, kpts1_norm, np.eye(3),
            method=cv2.RANSAC + cv2.LMEDS,  # Adding LMEDS for local optimization
            prob=0.999, threshold=norm_thresh)
    else:
        # Use standard RANSAC
        E, mask = cv2.findEssentialMat(
            kpts0_norm, kpts1_norm, np.eye(3),
            method=cv2.RANSAC,
            prob=0.999, threshold=norm_thresh)
    
    best_R, best_t, best_E = None, None, None
    if E is not None:
        best_num_inliers = 0
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0_norm, kpts1_norm, np.eye(3), 1e9, mask=mask
            )
            if n > best_num_inliers:
                best_num_inliers = n
                best_R = R
                best_t = t
                best_E = _E
    
    if best_R is None or best_t is None or best_E is None:
        return None, None, None
    
    return np.hstack([best_R, best_t]), mask.ravel(), best_E

def plot_ransac_analysis(all_metrics, output_dir='.'):
        """
        Create plots analyzing the relationship between number of evaluated pairs and RANSAC/LO-RANSAC AUCs.
        
        Args:
            all_metrics: List of metrics dictionaries for each sequence
            output_dir: Directory to save the plot files
        """
        import matplotlib.pyplot as plt
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract data for plotting
        pairs = [m['num_evaluated_pairs'] for m in all_metrics]

        # Plot RANSAC AUCs
        plt.figure(figsize=(10, 6))
        for thresh in [5, 10, 20]:
            aucs = [m['ransac_aucs'][f'auc_{thresh}'] for m in all_metrics if m['ransac_aucs']]
            plt.scatter(pairs, aucs, label=f'RANSAC {thresh}°', alpha=0.6)
        plt.xlabel('Number of Image Pairs')
        plt.ylabel('AUC (%)')
        plt.title('RANSAC Performance vs Number of Pairs')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / 'ransac_analysis.png')
        plt.close()

        # Plot LO-RANSAC AUCs
        plt.figure(figsize=(10, 6))
        for thresh in [5, 10, 20]:
            aucs = [m['loransac_aucs'][f'auc_{thresh}'] for m in all_metrics if m['loransac_aucs']]
            plt.scatter(pairs, aucs, label=f'LO-RANSAC {thresh}°', alpha=0.6)
        plt.xlabel('Number of Image Pairs')
        plt.ylabel('AUC (%)')
        plt.title('LO-RANSAC Performance vs Number of Pairs')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / 'loransac_analysis.png')
        plt.close()

def evaluate_sequence(npz_path, txt_path, extractor, matcher, device='cuda'):
    """Evaluate LightGlue matching on a single sequence."""
    try:
        # Load data
        sequence_data = np.load(npz_path)
        timestamps, intrinsics, poses = load_pose_data(txt_path)
        
        # Initialize metrics storage
        num_matches = []
        ransac_errors = []
        loransac_errors = []
        
        # Process first frame
        img0 = sequence_data[sequence_data.files[0]]
        H, W = img0.shape[:2]
        image0 = numpy_image_to_torch(img0)
        feats0 = extractor.extract(image0.to(device))
        
        # Process subsequent frames
        for i in tqdm(range(1, len(sequence_data.files))):
            # Extract features
            img_curr = sequence_data[sequence_data.files[i]]
            image1 = numpy_image_to_torch(img_curr)
            feats1 = extractor.extract(image1.to(device))
            
            # Match features
            matches01 = matcher({"image0": feats0, "image1": feats1})
            feats0_nobatch, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            
            # Get matched keypoints
            kpts0, kpts1 = feats0_nobatch["keypoints"], feats1["keypoints"]
            matches = matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
            
            # Convert to numpy
            m_kpts0 = m_kpts0.cpu().numpy()
            m_kpts1 = m_kpts1.cpu().numpy()
            num_matches.append(len(m_kpts0))
            
            if len(m_kpts0) >= 5:  # Minimum points for essential matrix
                # Setup camera intrinsics
                K1 = np.array([[intrinsics[0, 0]*W, 0, intrinsics[0, 2]*W],
                               [0, intrinsics[0, 1]*H, intrinsics[0, 3]*H],
                               [0, 0, 1]], dtype=np.float64)
                K2 = np.array([[intrinsics[i, 0]*W, 0, intrinsics[i, 2]*W],
                               [0, intrinsics[i, 1]*H, intrinsics[i, 3]*H],
                               [0, 0, 1]], dtype=np.float64)
                
                # Estimate pose with both RANSAC and LO-RANSAC
                T_ransac, mask_ransac, _ = estimate_pose(m_kpts0, m_kpts1, K1, K2, use_loransac=False)
                T_loransac, mask_loransac, _ = estimate_pose(m_kpts0, m_kpts1, K1, K2, use_loransac=True)
                
                if T_ransac is not None or T_loransac is not None:
                    # Get ground truth relative pose
                    pose0 = poses[0, :3, :]
                    pose_curr = poses[i, :3, :]
                    
                    # Compute relative pose
                    R0 = pose0[:, :3]
                    t0 = pose0[:, 3]
                    Ri = pose_curr[:, :3]
                    ti = pose_curr[:, 3]
                    
                    R_rel = Ri @ np.linalg.inv(R0)
                    t_rel = ti - R_rel @ t0
                    
                    T_gt = np.eye(4)
                    T_gt[:3, :3] = R_rel
                    T_gt[:3, 3] = t_rel
                    
                    # Compute errors for both methods
                    if T_ransac is not None:
                        R_err_ransac, _ = compute_pose_error(T_ransac, T_gt[:3, :])
                        ransac_errors.append(R_err_ransac)
                    
                    if T_loransac is not None:
                        R_err_loransac, _ = compute_pose_error(T_loransac, T_gt[:3, :])
                        loransac_errors.append(R_err_loransac)
        
        # Compute metrics
        if ransac_errors or loransac_errors:
            # Compute AUC metrics
            thresholds = [5, 10, 20]
            ransac_aucs = compute_auc(ransac_errors, thresholds) if ransac_errors else None
            loransac_aucs = compute_auc(loransac_errors, thresholds) if loransac_errors else None
            
            return {
                'mean_num_matches': np.mean(num_matches),
                'ransac_aucs': ransac_aucs,
                'loransac_aucs': loransac_aucs,
                'num_evaluated_pairs': len(sequence_data.files) - 1,
                'num_successful_ransac': len(ransac_errors),
                'num_successful_loransac': len(loransac_errors)
            }
        return None
            
    except Exception as e:
        print(f"Error processing sequence: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_dataset(dataset_path, extractor, matcher, num_sequences=None):
    """Evaluate LightGlue on multiple sequences."""
    dataset_path = Path(dataset_path)
    npz_path = dataset_path / 'realestate/test'
    txt_path = dataset_path / 'RealEstate10K/test'
    
    # Get all sequence paths
    sequences = list(npz_path.glob('*/data.npz'))
    if num_sequences is not None:
        sequences = sequences[:num_sequences]
    
    all_metrics = []
    for seq_path in tqdm(sequences, desc='Evaluating sequences'):
        seq_id = seq_path.parent.name
        txt_file = txt_path / f'{seq_id}.txt'
        
        if txt_file.exists():
            metrics = evaluate_sequence(seq_path, txt_file, extractor, matcher)
            if metrics is not None:
                metrics['sequence_id'] = seq_id
                all_metrics.append(metrics)


    # Compute aggregate metrics
    if all_metrics:
        # optional plot
        plot_ransac_analysis(all_metrics)
        aggregate_metrics = {
            'total_image_pairs': sum([m['num_evaluated_pairs'] for m in all_metrics]),
            'mean_num_matches': np.mean([m['mean_num_matches'] for m in all_metrics]),
            'ransac_aucs': {
                'auc_5': np.mean([m['ransac_aucs']['auc_5'] for m in all_metrics if m['ransac_aucs']]),
                'auc_10': np.mean([m['ransac_aucs']['auc_10'] for m in all_metrics if m['ransac_aucs']]),
                'auc_20': np.mean([m['ransac_aucs']['auc_20'] for m in all_metrics if m['ransac_aucs']])
            },
            'loransac_aucs': {
                'auc_5': np.mean([m['loransac_aucs']['auc_5'] for m in all_metrics if m['loransac_aucs']]),
                'auc_10': np.mean([m['loransac_aucs']['auc_10'] for m in all_metrics if m['loransac_aucs']]),
                'auc_20': np.mean([m['loransac_aucs']['auc_20'] for m in all_metrics if m['loransac_aucs']])
            },
            'num_sequences': len(all_metrics)
        }
        return aggregate_metrics, all_metrics
    return None, []

if __name__ == '__main__':
    # Set base dataset path
    dataset_path = '/media/liming/Base/datasets'
    device = 'cuda'
    
    # Initialize models
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # extractor = SIFT(max_num_keypoints=2048).eval().to(device)
    # matcher = LightGlue(features="sift").eval().to(device)
    
    # Run evaluation
    aggregate_metrics, sequence_metrics = evaluate_dataset(dataset_path, extractor, matcher, num_sequences=3)
    
    if aggregate_metrics is not None:
        print("\nAggregate Metrics:")
        print("\nRANSAC AUCs:")
        for k, v in aggregate_metrics['ransac_aucs'].items():
            print(f"{k}: {v:.2f}%")
        print("\nLO-RANSAC AUCs:")
        for k, v in aggregate_metrics['loransac_aucs'].items():
            print(f"{k}: {v:.2f}%")
        print(f"\nNumber of sequences evaluated: {aggregate_metrics['num_sequences']}")
        print(f"Mean number of matches: {aggregate_metrics['mean_num_matches']:.2f}")
        print(f"Total number of image pairs evaluated: {aggregate_metrics['total_image_pairs']}")