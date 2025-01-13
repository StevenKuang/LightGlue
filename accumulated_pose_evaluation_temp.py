import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from lightglue import SuperPoint, LightGlue, SIFT
from lightglue.utils import numpy_image_to_torch, rbd
import cv2
import traceback

class PoseMetrics:
    """Class to handle pose estimation metrics and analysis."""
    
    def __init__(self):
        # Initialize basic counters
        self.num_matches = []
        self.num_evaluated_pairs = 0
        self.stride = 1
        
        # Initialize error storage with separate translation metrics
        self.errors = {
            'ransac': {
                'R': [],
                't_angle': [],      # Angular difference between vectors (degrees)
                't_combined': [],   # Combined direction and scale error (0-1)
                'combined': []      # For AUC calculation
            },
            'loransac': {
                'R': [],
                't_angle': [],
                't_combined': [],
                'combined': []
            }
        }
        
        # Initialize progression storage for both metrics
        self.progression = {
            'ransac': {
                'R': [],
                't_angle': [],
                't_combined': []
            },
            'loransac': {
                'R': [],
                't_angle': [],
                't_combined': []
            }
        }

    
    def set_stride(self, stride):
        """Set the frame sampling stride."""
        self.stride = stride
    
    def add_match_count(self, count):
        """Add number of matches for current frame pair."""
        self.num_matches.append(count)
    
    def add_errors(self, method, R_err, t_err_angle, t_err_combined):
        """Add errors for given method with both translation metrics."""
        self.errors[method]['R'].append(R_err)
        self.errors[method]['t_angle'].append(t_err_angle)
        self.errors[method]['t_combined'].append(t_err_combined)
        
        # For AUC, scale t_combined from [0,1] to rough degree equivalent
        t_err_scaled = t_err_combined * 90  # Map [0,1] to [0,90] degrees
        combined_err = max(R_err, t_err_scaled)
        self.errors[method]['combined'].append(combined_err)
        
        # Update progression with both metrics
        self.progression[method]['R'].append(R_err)
        self.progression[method]['t_angle'].append(t_err_angle)
        self.progression[method]['t_combined'].append(t_err_combined)
    
    @property
    def mean_num_matches(self):
        """Calculate mean number of matches."""
        return np.mean(self.num_matches) if self.num_matches else 0
    
    def get_final_errors(self, method):
        """Get final errors for given method."""
        if not self.errors[method]['R']:
            return None, None
        return (self.errors[method]['R'][-1], 
                self.errors[method]['t'][-1])
    
    def compute_aucs(self, thresholds=[5, 10, 20]):
        """Compute AUCs for all methods."""
        aucs = {}
        for method in ['ransac', 'loransac']:
            if self.errors[method]['combined']:
                aucs[method] = {
                    f'auc_{thresh}': np.mean(
                        (np.array(self.errors[method]['combined']) <= thresh).astype(float)
                    ) * 100
                    for thresh in thresholds
                }
        return aucs
    
    def to_dict(self):
        """Convert metrics to dictionary format."""
        aucs = self.compute_aucs()
        return {
            'mean_num_matches': self.mean_num_matches,
            'num_evaluated_pairs': self.num_evaluated_pairs,
            'num_successful_ransac': len(self.errors['ransac']['combined']),
            'num_successful_loransac': len(self.errors['loransac']['combined']),
            'ransac_aucs': aucs.get('ransac'),
            'loransac_aucs': aucs.get('loransac'),
            'stride': self.stride,  # Add stride to output dictionary
            # Store progression data
            'ransac_R_progression': self.progression['ransac']['R'],
            'ransac_t_angle_progression': self.progression['ransac']['t_angle'],
            'ransac_t_combined_progression': self.progression['ransac']['t_combined'],
            'loransac_R_progression': self.progression['loransac']['R'],
            'loransac_t_angle_progression': self.progression['loransac']['t_angle'],
            'loransac_t_combined_progression': self.progression['loransac']['t_combined'],
            # Store final errors
            'final_ransac_R_error': self.errors['ransac']['R'][-1] if self.errors['ransac']['R'] else None,
            'final_ransac_t_error': self.errors['ransac']['t_combined'][-1] if self.errors['ransac']['t_combined'] else None,
            'final_loransac_R_error': self.errors['loransac']['R'][-1] if self.errors['loransac']['R'] else None,
            'final_loransac_t_error': self.errors['loransac']['t_combined'][-1] if self.errors['loransac']['t_combined'] else None
        }
class MetricsAggregator:
    """Class to aggregate metrics from multiple sequences."""
    
    def __init__(self):
        self.all_metrics = []
    
    def add_sequence_metrics(self, metrics_dict):
        """Add metrics from a single sequence."""
        if metrics_dict is not None:
            self.all_metrics.append(metrics_dict)
    
    def aggregate(self):
        """Compute aggregate metrics across all sequences."""
        if not self.all_metrics:
            return None
        
        return {
            'total_image_pairs': sum(m['num_evaluated_pairs'] for m in self.all_metrics),
            'mean_num_matches': np.mean([m['mean_num_matches'] for m in self.all_metrics]),
            'ransac_aucs': {
                'auc_5': np.mean([m['ransac_aucs']['auc_5'] for m in self.all_metrics if m['ransac_aucs']]),
                'auc_10': np.mean([m['ransac_aucs']['auc_10'] for m in self.all_metrics if m['ransac_aucs']]),
                'auc_20': np.mean([m['ransac_aucs']['auc_20'] for m in self.all_metrics if m['ransac_aucs']])
            },
            'loransac_aucs': {
                'auc_5': np.mean([m['loransac_aucs']['auc_5'] for m in self.all_metrics if m['loransac_aucs']]),
                'auc_10': np.mean([m['loransac_aucs']['auc_10'] for m in self.all_metrics if m['loransac_aucs']]),
                'auc_20': np.mean([m['loransac_aucs']['auc_20'] for m in self.all_metrics if m['loransac_aucs']])
            },
            'mean_final_ransac_R_error': np.mean([m['final_ransac_R_error'] for m in self.all_metrics if m['final_ransac_R_error'] is not None]),
            'mean_final_ransac_t_error': np.mean([m['final_ransac_t_error'] for m in self.all_metrics if m['final_ransac_t_error'] is not None]),
            'mean_final_loransac_R_error': np.mean([m['final_loransac_R_error'] for m in self.all_metrics if m['final_loransac_R_error'] is not None]),
            'mean_final_loransac_t_error': np.mean([m['final_loransac_t_error'] for m in self.all_metrics if m['final_loransac_t_error'] is not None]),
            'num_sequences': len(self.all_metrics)
        }
    
    def plot_error_progressions(self, output_dir='.', methods='both'):
        """
        Plot error progressions for all sequences considering frame stride.
        
        Args:
            output_dir: Directory to save plots
            methods: Which methods to plot. Can be 'both', 'r', 'lo', 
                    or a list containing any combination of ['r', 'lo']
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Process methods argument
        if methods == 'both':
            methods_to_plot = ['r', 'lo']
        elif isinstance(methods, str):
            methods = methods.lower()
            if methods not in ['r', 'lo']:
                raise ValueError("methods must be 'both', 'r', 'lo', or a list of these")
            methods_to_plot = [methods]
        elif isinstance(methods, (list, tuple)):
            methods_to_plot = [m.lower() for m in methods]
            if not all(m in ['r', 'lo'] for m in methods_to_plot):
                raise ValueError("All methods must be either 'r' or 'lo'")
        else:
            raise ValueError("methods must be string or list")

        # Method settings
        method_info = {
            'r': {'color': 'b', 'name': 'RANSAC', 'key': 'ransac'},
            'lo': {'color': 'r', 'name': 'LO-RANSAC', 'key': 'loransac'}
        }
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Helper function to create plots
        def make_plot(error_type, ylabel):
            plt.figure(figsize=(12, 6))
            for i, metrics in enumerate(self.all_metrics):
                stride = metrics.get('stride', 1)
                for m in methods_to_plot:
                    info = method_info[m]
                    prog_key = f"{info['key']}_{error_type}_progression"
                    if metrics[prog_key]:
                        frames = np.arange(0, len(metrics[prog_key]) * stride, stride)
                        plt.plot(frames, metrics[prog_key], 
                                f"{info['color']}-", alpha=0.2,
                                label=f"{info['name']} (s={stride})" if i == 0 else '')
            
            plt.xlabel('Frame Number')
            plt.ylabel(ylabel)
            plt.title(f'{error_type.capitalize()} Error Progression')
            if len(self.all_metrics) > 0:
                plt.legend()
            plt.grid(True)
            plt.savefig(Path(output_dir) / f'{error_type}_error_progression.png')
            plt.close()
        
        # Create both plots
        make_plot('R', 'Rotation Error (degrees)')
        make_plot('t_combined', 'Translation Error (degrees)')

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

def validate_camera_data(timestamps, intrinsics, poses, verbose=True):
    """
    Validate camera data from RealEstate10K format.
    
    Args:
        timestamps: Array of timestamps
        intrinsics: Nx6 array of intrinsic parameters 
        poses: Nx3x4 array of camera poses
        verbose: Whether to print validation messages
        
    Returns:
        is_valid: Boolean indicating if all checks passed
        messages: List of validation messages
    """
    messages = []
    is_valid = True
    
    # Check timestamps are monotonically increasing
    if not np.all(np.diff(timestamps) >= 0):
        messages.append("Warning: Timestamps are not monotonically increasing")
        is_valid = False
    
    # Check intrinsics are normalized (between 0 and 1)
    # focal_length_x, focal_length_y, principal_point_x, principal_point_y
    for i, name in enumerate(['fx', 'fy', 'cx', 'cy']):
        vals = intrinsics[:, i]
        if np.any(vals < 0) or np.any(vals > 10):  # Allow some margin for numerical errors
            messages.append(f"Warning: {name} values outside normalized range [0,1]")
            messages.append(f"Range: [{vals.min():.3f}, {vals.max():.3f}]")
            is_valid = False
            
    # Check rotation matrices are valid
    for i in range(len(poses)):
        R = poses[i, :3, :3]
        
        # Check orthogonality
        RtR = R.T @ R
        I = np.eye(3)
        if not np.allclose(RtR, I, atol=1e-3):
            messages.append(f"Warning: Non-orthogonal rotation matrix at frame {i}")
            messages.append(f"RtR error: {np.max(np.abs(RtR - I)):.3e}")
            is_valid = False
        
        # Check determinant is 1 (proper rotation)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-3):
            messages.append(f"Warning: Invalid rotation determinant at frame {i}: {det:.3f}")
            is_valid = False
            
    if verbose:
        if not is_valid:
            print("\n".join(messages))
            
    return is_valid, messages

def load_pose_data(txt_path, validate=True):
    """
    Load and validate camera poses and timestamps from RealEstate10K text file.
    
    Args:
        txt_path: Path to text file
        validate: Whether to run validation checks
    
    Returns:
        timestamps: Array of timestamps
        intrinsics: Array of camera intrinsics
        poses: Array of camera poses
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first line (header) and process the rest
    data_lines = [line.strip().split() for line in lines[1:]]
    data = np.array(data_lines)
    
    # Check we have the expected number of columns
    if data.shape[1] != 19:
        raise ValueError(f"Expected 19 columns but got {data.shape[1]}")
    
    timestamps = data[:, 0].astype(float)
    intrinsics = data[:, 1:7].astype(float)  # fx, fy, cx, cy, k1, k2
    poses = data[:, 7:].astype(float).reshape(-1, 3, 4)  # R|t
    
    if validate:
        is_valid, messages = validate_camera_data(timestamps, intrinsics, poses)
        if not is_valid:
            print("Note: Proceeding with invalid camera data")
            
    return timestamps, intrinsics, poses

def get_sequence_id(identifier, dataset_path):
    """
    Convert index or sequence ID to sequence ID.
    
    Args:
        identifier: Either an integer index or string sequence ID
        dataset_path: Base path to dataset
        
    Returns:
        sequence_id: String sequence ID
    """
    from pathlib import Path
    
    # If already a string ID, verify it exists
    if isinstance(identifier, str):
        sequence_path = Path(dataset_path) / 'realestate/test' / identifier
        if sequence_path.exists():
            return identifier
        raise ValueError(f"Sequence ID {identifier} not found")
    
    # If index, get the corresponding sequence ID
    if isinstance(identifier, int):
        sequences = sorted(p.parent.name for p in 
                         Path(dataset_path).glob('realestate/test/*/data.npz'))
        if 0 <= identifier < len(sequences):
            return sequences[identifier]
        raise ValueError(f"Index {identifier} out of range (0-{len(sequences)-1})")
    
    raise ValueError("Identifier must be string ID or integer index")

def list_sequences(dataset_path):
    """
    List all available sequences with their indices.
    
    Args:
        dataset_path: Base path to dataset
        
    Returns:
        List of (index, sequence_id) tuples
    """
    from pathlib import Path
    
    sequences = sorted(p.parent.name for p in 
                      Path(dataset_path).glob('realestate/test/*/data.npz'))
    return list(enumerate(sequences))

def play_sequence(identifier, dataset_path, metrics=None, delay=50, backend='matplotlib'):
    """
    Play a video sequence with optional error visualization.
    Supports both matplotlib and OpenCV backends.
    
    Args:
        identifier: Either an integer index or string sequence ID
        dataset_path: Base path to dataset
        metrics: Optional metrics dict for the sequence to overlay error information
        delay: Delay between frames in milliseconds (default: 50)
        backend: Display backend to use ('matplotlib' or 'opencv')
    """
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.style.use('dark_background')
    
    # Get sequence ID from identifier
    sequence_id = get_sequence_id(identifier, dataset_path)
    
    # Construct paths
    npz_path = Path(dataset_path) / 'realestate/test' / sequence_id / 'data.npz'
    
    if not npz_path.exists():
        print(f"Sequence data not found at {npz_path}")
        return
        
    # Load sequence data
    sequence_data = np.load(npz_path)
    frame_files = sorted(sequence_data.files)  # Ensure frames are in order
    
    # Get sequence metrics if available
    ransac_errors = []
    loransac_errors = []
    if metrics is not None:
        if 'ransac_R_progression' in metrics:
            ransac_errors = list(zip(metrics['ransac_R_progression'], 
                                   metrics['ransac_t_progression']))
        if 'loransac_R_progression' in metrics:
            loransac_errors = list(zip(metrics['loransac_R_progression'], 
                                     metrics['loransac_t_progression']))

    if backend == 'matplotlib':
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Initialize plot elements with first frame to get correct dimensions
        first_frame = sequence_data[frame_files[0]]
        im = ax.imshow(first_frame)
        ax.set_xticks([])  # Hide axis ticks
        ax.set_yticks([])
        
        text_ransac = ax.text(0.02, 0.95, '', color='red', transform=ax.transAxes)
        text_loransac = ax.text(0.02, 0.90, '', color='green', transform=ax.transAxes)
        text_frame = ax.text(0.02, 0.05, '', color='white', transform=ax.transAxes)
        
        # Animation update function
        def update(frame_idx):
            frame = sequence_data[frame_files[frame_idx]]
            
            # Update image
            im.set_array(frame)
            if frame_idx == 0:
                im.axes.set_xlim(0, frame.shape[1])
                im.axes.set_ylim(frame.shape[0], 0)
            
            # Update text
            text_frame.set_text(f'Frame: {frame_idx}')
            
            if frame_idx > 0:
                if frame_idx-1 < len(ransac_errors):
                    r_err, t_err = ransac_errors[frame_idx-1]
                    text_ransac.set_text(f'RANSAC - R: {r_err:.2f}°, t: {t_err:.2f}°')
                
                if frame_idx-1 < len(loransac_errors):
                    r_err, t_err = loransac_errors[frame_idx-1]
                    text_loransac.set_text(f'LO-RANSAC - R: {r_err:.2f}°, t: {t_err:.2f}°')
            
            return [im, text_ransac, text_loransac, text_frame]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=len(frame_files),
            interval=delay, blit=True
        )
        
        # Add title with both index and ID
        try:
            idx = [s[1] for s in list_sequences(dataset_path)].index(sequence_id)
            plt.title(f'Sequence {idx} (ID: {sequence_id})')
        except ValueError:
            plt.title(f'Sequence ID: {sequence_id}')
        
        # Show plot with playback controls
        plt.show()
        
    elif backend == 'opencv':
        try:
            import cv2
            
            # Create window
            window_name = f'Sequence {sequence_id}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            for i, frame_file in enumerate(frame_files):
                frame = sequence_data[frame_file]
                
                # Add error information
                if i > 0:
                    text_y = 30
                    if i-1 < len(ransac_errors):
                        r_err, t_err = ransac_errors[i-1]
                        cv2.putText(frame, f'RANSAC - R: {r_err:.2f}°, t: {t_err:.2f}°',
                                  (10, text_y), font, font_scale, (0, 0, 255), thickness)
                        text_y += 25
                        
                    if i-1 < len(loransac_errors):
                        r_err, t_err = loransac_errors[i-1]
                        cv2.putText(frame, f'LO-RANSAC - R: {r_err:.2f}°, t: {t_err:.2f}°',
                                  (10, text_y), font, font_scale, (0, 255, 0), thickness)
                
                cv2.putText(frame, f'Frame: {i}', (10, frame.shape[0]-20),
                           font, font_scale, (255, 255, 255), thickness)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            
        except (ImportError, cv2.error) as e:
            print(f"OpenCV error: {str(e)}")
            print("Falling back to matplotlib backend...")
            play_sequence(identifier, dataset_path, metrics, delay, backend='matplotlib')
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
def find_sequences_with_spikes(sequence_metrics, top_n=5, default_stride=1):
    """
    Find sequences with error spikes in rotation and/or translation.
    Separates RANSAC and LO-RANSAC results.
    Each sequence appears only once per error type and method.
    
    Args:
        sequence_metrics: List of metrics dictionaries for each sequence
        top_n: Number of top sequences to return for each type and method
        default_stride: Default stride value if not found in metrics
    
    Returns:
        Nested dictionary containing results for each method (RANSAC/LO-RANSAC),
        and within each method, lists of sequences for rotation and translation errors
    """
    import numpy as np
    from collections import defaultdict
    
    # Initialize nested structure for results
    results = {
        'ransac': {
            'rotation': defaultdict(lambda: {'error': -float('inf')}),
            'translation': defaultdict(lambda: {'error': -float('inf')})
        },
        'loransac': {
            'rotation': defaultdict(lambda: {'error': -float('inf')}),
            'translation': defaultdict(lambda: {'error': -float('inf')})
        }
    }
    
    # Process each sequence
    for metrics in sequence_metrics:
        seq_id = metrics.get('sequence_id')
        if not seq_id:
            continue
        
        stride = metrics.get('stride', default_stride)
        
        # Process RANSAC results
        if 'ransac_R_progression' in metrics and metrics['ransac_R_progression']:
            r_errors = metrics['ransac_R_progression']
            max_r = max(r_errors)
            max_r_idx = r_errors.index(max_r)
            actual_frame_idx = max_r_idx * stride
            
            results['ransac']['rotation'][seq_id] = {
                'error': max_r,
                'frame_idx': actual_frame_idx,
                'stride': stride
            }
            
        if 'ransac_t_progression' in metrics and metrics['ransac_t_progression']:
            t_errors = metrics['ransac_t_progression']
            max_t = max(t_errors)
            max_t_idx = t_errors.index(max_t)
            actual_frame_idx = max_t_idx * stride
            
            results['ransac']['translation'][seq_id] = {
                'error': max_t,
                'frame_idx': actual_frame_idx,
                'stride': stride
            }
            
        # Process LO-RANSAC results
        if 'loransac_R_progression' in metrics and metrics['loransac_R_progression']:
            r_errors = metrics['loransac_R_progression']
            max_r = max(r_errors)
            max_r_idx = r_errors.index(max_r)
            actual_frame_idx = max_r_idx * stride
            
            results['loransac']['rotation'][seq_id] = {
                'error': max_r,
                'frame_idx': actual_frame_idx,
                'stride': stride
            }
            
        if 'loransac_t_progression' in metrics and metrics['loransac_t_progression']:
            t_errors = metrics['loransac_t_progression']
            max_t = max(t_errors)
            max_t_idx = t_errors.index(max_t)
            actual_frame_idx = max_t_idx * stride
            
            results['loransac']['translation'][seq_id] = {
                'error': max_t,
                'frame_idx': actual_frame_idx,
                'stride': stride
            }
    
    # Convert to sorted lists
    final_results = {
        'ransac': {
            'rotation': [],
            'translation': []
        },
        'loransac': {
            'rotation': [],
            'translation': []
        }
    }
    
    # Process and sort each category
    for method in ['ransac', 'loransac']:
        for error_type in ['rotation', 'translation']:
            # Convert dictionary to list of tuples
            error_list = [
                (seq_id, data['error'], data['frame_idx'], data['stride'])
                for seq_id, data in results[method][error_type].items()
                if data['error'] > -float('inf')  # Filter out unprocessed sequences
            ]
            
            # Sort by error magnitude and take top N
            final_results[method][error_type] = sorted(
                error_list,
                key=lambda x: x[1],  # Sort by error value
                reverse=True
            )[:top_n]
    
    return final_results

def compute_pose_error(T_pred, T_gt, ignore_gt_t_thr=0.0, eps=1e-10):
    """
    Compute both angle-based and combined translation error metrics.
    
    Returns:
        R_error: rotation error in degrees
        t_error_angle: translation error in degrees (angular difference)
        t_error_combined: translation error as normalized combined metric (0-1)
    """
    # Extract rotation and translation
    R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    
    # Rotation error (degrees)
    cos_r = np.clip((np.trace(R_pred @ R_gt.T) - 1) / 2, -1.0, 1.0)
    R_error = np.rad2deg(np.abs(np.arccos(cos_r)))
    
    # Handle pure rotation case
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:
        return R_error, 0.0, 0.0
        
    # Compute angle-based translation error
    n = np.clip(np.linalg.norm(t_pred) * np.linalg.norm(t_gt), eps, None)
    cos_t = np.clip(np.dot(t_pred, t_gt) / n, -1.0, 1.0)
    t_error_angle = np.rad2deg(np.arccos(cos_t))
    # Handle essential matrix ambiguity
    t_error_angle = min(t_error_angle, 180 - t_error_angle)
    
    # Compute combined translation error
    t_pred_norm = t_pred / (np.linalg.norm(t_pred) + eps)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + eps)
    
    # Direction error (0 to 1, where 0 is perfect alignment)
    dir_error = 0.5 * (1 - np.dot(t_pred_norm, t_gt_norm))
    
    # Scale error (0 to 1, where 0 means same scale)
    scale_ratio = np.linalg.norm(t_pred) / (np.linalg.norm(t_gt) + eps)
    if scale_ratio > 1:
        scale_ratio = 1 / scale_ratio
    scale_error = 1 - scale_ratio
    
    # Combined metric (0 to 1)
    t_error_combined = 0.5 * (dir_error + scale_error)
    
    return R_error, t_error_angle, t_error_combined

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
    """Create plots analyzing RANSAC/LO-RANSAC performance and error progression."""
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot final drift vs number of pairs
    plt.figure(figsize=(10, 6))
    pairs = [m['num_evaluated_pairs'] for m in all_metrics]
    final_ransac_errors = [m['final_ransac_error'] for m in all_metrics if m['final_ransac_error'] is not None]
    final_loransac_errors = [m['final_loransac_error'] for m in all_metrics if m['final_loransac_error'] is not None]
    
    if final_ransac_errors:
        plt.scatter(pairs, final_ransac_errors, color='#e41a1c', label='RANSAC', alpha=0.6)  # Red
    if final_loransac_errors:
        plt.scatter(pairs, final_loransac_errors, color='#984ea3', label='LO-RANSAC', alpha=0.6)  # Purple
    
    plt.xlabel('Number of Image Pairs')
    plt.ylabel('Final Drift (degrees)')
    plt.title('Final Pose Drift vs Number of Pairs')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'final_drift_analysis.png')
    plt.close()

    # Plot error progression for each sequence
    plt.figure(figsize=(12, 6))
    for i, metrics in enumerate(all_metrics):
        if metrics['ransac_error_progression']:
            frames = range(len(metrics['ransac_error_progression']))
            plt.plot(frames, metrics['ransac_error_progression'], 
                    'b-', alpha=0.2, label='RANSAC' if i == 0 else '')
        if metrics['loransac_error_progression']:
            frames = range(len(metrics['loransac_error_progression']))
            plt.plot(frames, metrics['loransac_error_progression'], 
                    'r-', alpha=0.2, label='LO-RANSAC' if i == 0 else '')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Cumulative Error (degrees)')
    plt.title('Error Progression Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'error_progression.png')
    plt.close()

def compare_translation_metrics(sequence_metrics, output_dir='.', methods=['ransac', 'loransac']):
    """
    Create side-by-side visualizations of two translation error metrics.
    
    Args:
        sequence_metrics: List of metrics dictionaries for each sequence
        output_dir: Directory to save visualization plots
        methods: List of pose estimation methods to compare
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Setup plot style
    plt.style.use('classic')
    plt.rc('font', size=10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = {'ransac': '#1f77b4', 'loransac': '#d62728'}
    
    # Get the stride value safely - use the first valid sequence or default to 1
    stride = 1  # Default value
    for sequence in sequence_metrics:
        if 'stride' in sequence:
            stride = sequence['stride']
            break
    
    # Plot both metrics
    for sequence in sequence_metrics:
        current_stride = sequence.get('stride', stride)  # Use sequence-specific stride if available
        for method in methods:
            # Plot angle-based metric
            key_angle = f'{method}_t_angle_progression'
            if key_angle in sequence and sequence[key_angle]:  # Check if data exists
                frames = np.arange(0, len(sequence[key_angle]) * current_stride, current_stride)
                ax1.plot(frames, sequence[key_angle],
                        color=colors[method], alpha=0.2,
                        label=method.upper() if sequence == sequence_metrics[0] else '')
            
            # Plot combined metric
            key_combined = f'{method}_t_combined_progression'
            if key_combined in sequence and sequence[key_combined]:  # Check if data exists
                frames = np.arange(0, len(sequence[key_combined]) * current_stride, current_stride)
                ax2.plot(frames, sequence[key_combined],
                        color=colors[method], alpha=0.2,
                        label=method.upper() if sequence == sequence_metrics[0] else '')
    
    # Configure plots
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Translation Error (degrees)')
    ax1.set_title('Angle-based Translation Error\n(Direction Only)')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Translation Error (normalized)')
    ax2.set_title('Combined Translation Error\n(Direction + Scale)')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Add overall title with safe stride value
    fig.suptitle(f'Translation Error Metrics Comparison (stride={stride})', y=1.02)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.savefig(Path(output_dir) / 'translation_metrics_comparison.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def visualize_error_distributions(sequence_metrics, output_dir='.', methods=['ransac', 'loransac']):
    """Create histograms showing the distribution of both translation error metrics."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    plt.style.use('classic')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = {'ransac': '#1f77b4', 'loransac': '#d62728'}
    
    for method in methods:
        # Collect both types of errors
        angle_errors = []
        combined_errors = []
        
        for sequence in sequence_metrics:
            if f'{method}_t_angle_progression' in sequence:
                angle_errors.extend(sequence[f'{method}_t_angle_progression'])
            if f'{method}_t_combined_progression' in sequence:
                combined_errors.extend(sequence[f'{method}_t_combined_progression'])
        
        if angle_errors:
            ax1.hist(angle_errors, bins=50, alpha=0.5,
                    color=colors[method], label=method.upper(),
                    density=True, edgecolor='black', linewidth=0.5)
        
        if combined_errors:
            ax2.hist(combined_errors, bins=50, alpha=0.5,
                    color=colors[method], label=method.upper(),
                    density=True, edgecolor='black', linewidth=0.5)
    
    # Configure plots
    ax1.set_xlabel('Translation Error (degrees)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Angle-based Translation Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.7)
    
    ax2.set_xlabel('Translation Error (normalized)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Combined Translation Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.7)
    
    plt.suptitle('Translation Error Distributions', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'translation_error_distributions.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_sequence(npz_path, txt_path, extractor, matcher, stride=1, device='cuda'):
    """
    Evaluate LightGlue matching on a single sequence with cumulative pose error.
    
    Args:
        npz_path: Path to sequence data npz file
        txt_path: Path to pose data txt file
        extractor: Feature extractor model
        matcher: Feature matcher model
        stride: Interval between frames (default: 1 for consecutive frames)
        device: Computation device
    """
    try:
        # Load data
        sequence_data = np.load(npz_path)
        timestamps, intrinsics, poses = load_pose_data(txt_path)
        
        # Get valid frame indices based on stride
        total_frames = len(sequence_data.files)
        frame_indices = range(0, total_frames, stride)
        
        # Initialize metrics tracker
        metrics = PoseMetrics()
        metrics.set_stride(stride)
        metrics.num_evaluated_pairs = len(frame_indices) - 1
        
        # Initialize cumulative poses with first frame's ground truth pose
        first_idx = frame_indices[0]
        T_gt_init = np.eye(4)
        T_gt_init[:3, :3] = poses[first_idx, :3, :3]
        T_gt_init[:3, 3] = poses[first_idx, :3, 3]
        cum_poses = {
            'ransac': T_gt_init.copy(),
            'loransac': T_gt_init.copy()
        }
        
        # Process first frame
        img_prev = sequence_data[sequence_data.files[first_idx]]
        H, W = img_prev.shape[:2]
        feats_prev = extractor.extract(numpy_image_to_torch(img_prev).to(device))
        
        # Process subsequent frames with stride
        for i in tqdm(range(1, len(frame_indices)), desc=f'Processing frames (stride={stride})'):
            curr_idx = frame_indices[i]
            prev_idx = frame_indices[i-1]
            
            # Extract features for current frame
            img_curr = sequence_data[sequence_data.files[curr_idx]]
            feats_curr = extractor.extract(numpy_image_to_torch(img_curr).to(device))
            feats_curr_raw = feats_curr.copy()
            
            # Match features
            matches = matcher({"image0": feats_prev, "image1": feats_curr})
            feats_prev_nobatch, feats_curr, matches = [rbd(x) for x in [feats_prev, feats_curr, matches]]
            
            # Get matched keypoints
            kpts_prev = feats_prev_nobatch["keypoints"]
            kpts_curr = feats_curr["keypoints"]
            matches_idx = matches["matches"]
            m_kpts_prev = kpts_prev[matches_idx[..., 0]].cpu().numpy()
            m_kpts_curr = kpts_curr[matches_idx[..., 1]].cpu().numpy()
            
            # Record number of matches
            metrics.add_match_count(len(m_kpts_prev))
            
            if len(m_kpts_prev) >= 5:  # Minimum points for essential matrix
                # Setup camera intrinsics
                K_prev = np.array([[intrinsics[prev_idx, 0]*W, 0, intrinsics[prev_idx, 2]*W],
                                 [0, intrinsics[prev_idx, 1]*H, intrinsics[prev_idx, 3]*H],
                                 [0, 0, 1]], dtype=np.float64)
                K_curr = np.array([[intrinsics[curr_idx, 0]*W, 0, intrinsics[curr_idx, 2]*W],
                                 [0, intrinsics[curr_idx, 1]*H, intrinsics[curr_idx, 3]*H],
                                 [0, 0, 1]], dtype=np.float64)
                
                # Get current ground truth pose
                T_gt_curr = np.eye(4)
                T_gt_curr[:3, :3] = poses[curr_idx, :3, :3]
                T_gt_curr[:3, 3] = poses[curr_idx, :3, 3]
                
                # Process both RANSAC methods
                for method, use_lo in [('ransac', False), ('loransac', True)]:
                    T_est, mask, _ = estimate_pose(m_kpts_prev, m_kpts_curr, 
                                                 K_prev, K_curr, 
                                                 use_loransac=use_lo)
                    
                    if T_est is not None:
                        # Accumulate pose
                        T_est_4x4 = np.eye(4)
                        T_est_4x4[:3, :] = T_est
                        cum_poses[method] = cum_poses[method] @ T_est_4x4
                        
                        # Compute and store both types of errors
                        R_err, t_err_angle, t_err_combined = compute_pose_error(
                            cum_poses[method], T_gt_curr
                        )
                        metrics.add_errors(method, R_err, t_err_angle, t_err_combined)
            
            # Update previous frame features
            feats_prev = feats_curr_raw
        
        # Return metrics if any poses were successfully estimated
        metrics_dict = metrics.to_dict() if (metrics.errors['ransac']['combined'] or 
                                           metrics.errors['loransac']['combined']) else None
        if metrics_dict:
            metrics_dict['stride'] = stride  # Add stride information to metrics
        return metrics_dict
            
    except Exception as e:
        print(f"Error processing sequence: {str(e)}")
        traceback.print_exc()
        return None

def evaluate_dataset(dataset_path, extractor, matcher, stride=1, num_sequences=None):
    """
    Evaluate LightGlue on multiple sequences.
    
    Args:000c3ab189999a83
        dataset_path: Path to dataset
        extractor: Feature extractor model
        matcher: Feature matcher model
        stride: Interval between frames (default: 1)
        num_sequences: Number of sequences to evaluate (optional)
    """
    dataset_path = Path(dataset_path)
    npz_path = dataset_path / 'realestate/test'
    txt_path = dataset_path / 'RealEstate10K/test'
    
    # Get sequence paths
    sequences = list(npz_path.glob('*/data.npz'))
    if num_sequences is not None:
        sequences = sequences[:num_sequences]
    
    # Initialize metrics aggregator
    aggregator = MetricsAggregator()
    
    # Process each sequence
    for seq_path in tqdm(sequences, desc='Evaluating sequences'):
        seq_id = seq_path.parent.name
        txt_file = txt_path / f'{seq_id}.txt'
        
        if txt_file.exists():
            metrics = evaluate_sequence(seq_path, txt_file, extractor, matcher, stride=stride)
            if metrics is not None:
                metrics['sequence_id'] = seq_id
                aggregator.add_sequence_metrics(metrics)
    
    # Return aggregated results
    if aggregator.all_metrics:
        plot_dir = f'plots_stride_{stride}'  # Add stride to plot directory name
        aggregator.plot_error_progressions(output_dir=plot_dir)
        return aggregator.aggregate(), aggregator.all_metrics
    return None, []

if __name__ == '__main__':
    # Set parameters
    dataset_path = '/media/liming/Base/datasets'
    device = 'cuda'
    stride = 5  # Set frame sampling stride (1 for consecutive frames)
    num_sequences = 10
    
    # play_sequence('4221bc1d4aea1a02', dataset_path, backend='opencv')
    # play_sequence('ae38058a530efbc3', dataset_path, backend='opencv')
    # play_sequence('4227369e7d0e735a', dataset_path, backend='opencv')
    # play_sequence(3, dataset_path, backend='opencv')

    # Initialize models
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    # extractor = SIFT(max_num_keypoints=2048).eval().to(device)
    # matcher = LightGlue(features="sift").eval().to(device)
    
    # Run evaluation
    print(f"\nEvaluating with frame stride: {stride}")
    aggregate_metrics, sequence_metrics = evaluate_dataset(
        dataset_path, extractor, matcher, 
        stride=stride, num_sequences=num_sequences
    )
    
    # Print results
    if aggregate_metrics:
        print("\nAggregate Metrics:")
        print(f"Frame stride: {stride}")
        print("\nRANSAC AUCs:")
        for k, v in aggregate_metrics['ransac_aucs'].items():
            print(f"{k}: {v:.2f}%")
            
        print("\nLO-RANSAC AUCs:")
        for k, v in aggregate_metrics['loransac_aucs'].items():
            print(f"{k}: {v:.2f}%")
            
        print(f"\nNumber of sequences evaluated: {aggregate_metrics['num_sequences']}")
        print(f"Total image pairs: {aggregate_metrics['total_image_pairs']}")
        print(f"Mean matches per pair: {aggregate_metrics['mean_num_matches']:.2f}")
        
        print("\nFinal Errors (degrees):")
        print(f"RANSAC - Rotation: {aggregate_metrics['mean_final_ransac_R_error']:.2f}")
        print(f"RANSAC - Translation: {aggregate_metrics['mean_final_ransac_t_error']:.2f}")
        print(f"LO-RANSAC - Rotation: {aggregate_metrics['mean_final_loransac_R_error']:.2f}")
        print(f"LO-RANSAC - Translation: {aggregate_metrics['mean_final_loransac_t_error']:.2f}")


    # Find top 5 error spikes for both RANSAC methods
    spikes = find_sequences_with_spikes(sequence_metrics, top_n=5)

    # Print RANSAC rotation errors
    print("\nRANSAC top rotation error spikes:")
    for seq_id, max_error, frame_idx, stride in spikes['ransac']['rotation']:
        print(f"Sequence {seq_id}: {max_error:.2f}° at frame {frame_idx} (stride={stride})")

    # Print LO-RANSAC rotation errors
    print("\nLO-RANSAC top rotation error spikes:")
    for seq_id, max_error, frame_idx, stride in spikes['loransac']['rotation']:
        print(f"Sequence {seq_id}: {max_error:.2f}° at frame {frame_idx} (stride={stride})")

    # Print RANSAC translation errors
    print("\nRANSAC top translation error spikes:")
    for seq_id, max_error, frame_idx, stride in spikes['ransac']['translation']:
        print(f"Sequence {seq_id}: {max_error:.2f}° at frame {frame_idx} (stride={stride})")

    # Print LO-RANSAC translation errors
    print("\nLO-RANSAC top translation error spikes:")
    for seq_id, max_error, frame_idx, stride in spikes['loransac']['translation']:
        print(f"Sequence {seq_id}: {max_error:.2f}° at frame {frame_idx} (stride={stride})")

    # After running evaluation
    compare_translation_metrics(sequence_metrics, output_dir='plots')
    visualize_error_distributions(sequence_metrics, output_dir='plots')

    # play_sequence(seq_id, dataset_path, metrics=sequence_metrics[get_sequence_id(seq_id)], backend='opencv')
