#!/usr/bin/env python3
"""
B7: Trajectory Reconstruction Analysis

Reconstructs pedestrian movement paths using RSSI-based distance estimates
and compares with ground truth positions to assess tracking accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class TrajectoryReconstructionAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.los_path = self.dataset_path / 'Deployment Distance' / 'los'
        self.nlos_path = self.dataset_path / 'Deployment Distance' / 'nlos'
        self.distances = [3, 5, 7, 9]
        self.positions = ['start', 'mid_facing', 'center', 'mid_away', 'end']

        # Position mapping for nLoS files
        self.position_map_nlos = {
            'start': 'start',
            'mid_facing': 'mid facing',
            'center': 'centre',
            'mid_away': 'mid away',
            'end': 'end'
        }

        # Reference RSSI at 1m for distance estimation
        # From A1 analysis: typical RSSI at 1m is around -50 dBm
        self.rssi_ref = -50
        # From A1: path loss exponent
        self.n_los = 0.735
        self.n_nlos = 0.415

    def load_data(self, orientation='los'):
        """Load RSSI data from Deployment Distance dataset"""
        data = []
        base_path = self.los_path if orientation == 'los' else self.nlos_path

        for pos in self.positions:
            # Adjust position name for nLoS
            if orientation == 'nlos':
                pos_dir = self.position_map_nlos.get(pos, pos)
            else:
                pos_dir = pos

            pos_path = base_path / pos_dir

            if not pos_path.exists():
                continue

            for run in range(1, 4):  # 3 runs
                run_path = pos_path / f'run{run}'

                if not run_path.exists():
                    continue

                # Load all distance files for this position/run
                for dist in self.distances:
                    try:
                        # File naming: {position}_{dist}m_run{run}.csv
                        if orientation == 'los':
                            data_file = run_path / f'{pos}_{dist}m_run{run}.csv'
                        else:
                            data_file = run_path / f'{pos_dir}_{dist}m_run{run}.csv'

                        if data_file.exists():
                            df = pd.read_csv(data_file)

                            if not df.empty and 'rssi' in df.columns:
                                # Add metadata
                                df['distance'] = dist
                                df['position'] = pos
                                df['orientation'] = orientation
                                df['run'] = run

                                # Convert timestamp (nanoseconds to datetime)
                                if 'time' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['time'], unit='ns')
                                elif 'timestamp' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                                data.append(df)
                    except Exception as e:
                        continue

        if data:
            return pd.concat(data, ignore_index=True)
        return pd.DataFrame()

    def estimate_distance_from_rssi(self, rssi, orientation='los'):
        """Estimate distance using log-distance path loss model"""
        n = self.n_los if orientation == 'los' else self.n_nlos
        # d = 10^((RSSI_ref - RSSI) / (10*n))
        estimated = 10 ** ((self.rssi_ref - rssi) / (10 * n))
        # Clip to reasonable range
        return np.clip(estimated, 0.5, 20)

    def reconstruct_trajectories(self, df):
        """Reconstruct movement trajectories from RSSI data"""
        trajectories = []

        for (pos, orient, run), group in df.groupby(['position', 'orientation', 'run']):
            # Sort by distance to create trajectory
            trajectory_points = []

            for dist in sorted(group['distance'].unique()):
                dist_data = group[group['distance'] == dist]

                if not dist_data.empty:
                    # Calculate statistics
                    mean_rssi = dist_data['rssi'].mean()
                    std_rssi = dist_data['rssi'].std()
                    median_rssi = dist_data['rssi'].median()

                    # Estimate distance from RSSI
                    estimated_dist = self.estimate_distance_from_rssi(mean_rssi, orient)

                    # Calculate error
                    error = estimated_dist - dist

                    trajectory_points.append({
                        'true_distance': dist,
                        'estimated_distance': estimated_dist,
                        'mean_rssi': mean_rssi,
                        'median_rssi': median_rssi,
                        'std_rssi': std_rssi,
                        'error': error,
                        'abs_error': abs(error),
                        'relative_error': abs(error) / dist * 100
                    })

            if trajectory_points:
                traj_df = pd.DataFrame(trajectory_points)
                traj_df['position'] = pos
                traj_df['orientation'] = orient
                traj_df['run'] = run
                trajectories.append(traj_df)

        if trajectories:
            return pd.concat(trajectories, ignore_index=True)
        return pd.DataFrame()

    def analyze_tracking_accuracy(self, trajectories_df):
        """Analyze accuracy of trajectory reconstruction"""
        analysis = {
            'overall': {
                'mean_error': trajectories_df['error'].mean(),
                'mean_abs_error': trajectories_df['abs_error'].mean(),
                'rmse': np.sqrt((trajectories_df['error']**2).mean()),
                'mean_relative_error': trajectories_df['relative_error'].mean()
            },
            'by_distance': {},
            'by_orientation': {},
            'by_position': {}
        }

        # By distance
        for dist in sorted(trajectories_df['true_distance'].unique()):
            dist_data = trajectories_df[trajectories_df['true_distance'] == dist]
            analysis['by_distance'][dist] = {
                'mae': dist_data['abs_error'].mean(),
                'std': dist_data['error'].std(),
                'relative_error': dist_data['relative_error'].mean()
            }

        # By orientation
        for orient in trajectories_df['orientation'].unique():
            orient_data = trajectories_df[trajectories_df['orientation'] == orient]
            analysis['by_orientation'][orient] = {
                'mae': orient_data['abs_error'].mean(),
                'std': orient_data['error'].std(),
                'relative_error': orient_data['relative_error'].mean()
            }

        # By position
        for pos in trajectories_df['position'].unique():
            pos_data = trajectories_df[trajectories_df['position'] == pos]
            analysis['by_position'][pos] = {
                'mae': pos_data['abs_error'].mean(),
                'std': pos_data['error'].std(),
                'relative_error': pos_data['relative_error'].mean()
            }

        return analysis

    def plot_trajectory_reconstruction(self, trajectories_df, output_dir):
        """Create comprehensive trajectory visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('B7: Trajectory Reconstruction Analysis', fontsize=16, fontweight='bold')

        # 1. True vs Estimated Distance
        ax = axes[0, 0]
        for orient in ['los', 'nlos']:
            data = trajectories_df[trajectories_df['orientation'] == orient]
            ax.scatter(data['true_distance'], data['estimated_distance'],
                      alpha=0.5, label=f'{orient.upper()}', s=50)

        # Perfect estimation line
        max_dist = trajectories_df['true_distance'].max()
        ax.plot([0, max_dist], [0, max_dist], 'r--', linewidth=2, label='Perfect Estimate')
        ax.set_xlabel('True Distance (m)', fontsize=11)
        ax.set_ylabel('Estimated Distance (m)', fontsize=11)
        ax.set_title('Distance Estimation Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max_dist + 1])
        ax.set_ylim([0, max(max_dist + 1, trajectories_df['estimated_distance'].max())])

        # 2. Tracking Error by Distance
        ax = axes[0, 1]
        distances = sorted(trajectories_df['true_distance'].unique())
        for orient in ['los', 'nlos']:
            errors = [trajectories_df[
                (trajectories_df['true_distance'] == d) &
                (trajectories_df['orientation'] == orient)
            ]['abs_error'].mean() for d in distances]
            ax.plot(distances, errors, marker='o', label=f'{orient.upper()}', linewidth=2)

        ax.set_xlabel('True Distance (m)', fontsize=11)
        ax.set_ylabel('Mean Absolute Error (m)', fontsize=11)
        ax.set_title('Tracking Error vs Distance', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error Distribution
        ax = axes[0, 2]
        errors_los = trajectories_df[trajectories_df['orientation'] == 'los']['error'].values
        errors_nlos = trajectories_df[trajectories_df['orientation'] == 'nlos']['error'].values

        ax.hist(errors_los, bins=20, alpha=0.6, label='LoS', edgecolor='black')
        ax.hist(errors_nlos, bins=20, alpha=0.6, label='nLoS', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Estimation Error (m)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Trajectory Examples - LoS
        ax = axes[1, 0]
        # Show one example run for each position
        for pos in ['start', 'center', 'end']:
            data = trajectories_df[
                (trajectories_df['position'] == pos) &
                (trajectories_df['orientation'] == 'los') &
                (trajectories_df['run'] == 1)
            ].sort_values('true_distance')

            if not data.empty:
                ax.plot(data['true_distance'], data['estimated_distance'],
                       marker='o', label=f'{pos}', linewidth=2)

        ax.plot([0, 10], [0, 10], 'k--', linewidth=1, alpha=0.5, label='Perfect')
        ax.set_xlabel('True Distance (m)', fontsize=11)
        ax.set_ylabel('Estimated Distance (m)', fontsize=11)
        ax.set_title('LoS Trajectory Examples', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Position-Specific Accuracy
        ax = axes[1, 1]
        positions = trajectories_df['position'].unique()
        mae_by_pos = [trajectories_df[trajectories_df['position'] == p]['abs_error'].mean()
                     for p in positions]

        ax.bar(range(len(positions)), mae_by_pos, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, rotation=45, ha='right')
        ax.set_ylabel('Mean Absolute Error (m)', fontsize=11)
        ax.set_title('Accuracy by Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Relative Error
        ax = axes[1, 2]
        for orient in ['los', 'nlos']:
            rel_errors = [trajectories_df[
                (trajectories_df['true_distance'] == d) &
                (trajectories_df['orientation'] == orient)
            ]['relative_error'].mean() for d in distances]
            ax.plot(distances, rel_errors, marker='o', label=f'{orient.upper()}', linewidth=2)

        ax.set_xlabel('True Distance (m)', fontsize=11)
        ax.set_ylabel('Relative Error (%)', fontsize=11)
        ax.set_title('Relative Tracking Error', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_reconstruction.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'trajectory_reconstruction.pdf', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved trajectory reconstruction plots")

    def plot_rssi_vs_distance(self, trajectories_df, output_dir):
        """Plot RSSI signal strength vs distance"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('RSSI Signal Propagation', fontsize=16, fontweight='bold')

        # LoS
        ax = axes[0]
        los_data = trajectories_df[trajectories_df['orientation'] == 'los']
        for pos in self.positions:
            pos_data = los_data[los_data['position'] == pos]
            if not pos_data.empty:
                ax.scatter(pos_data['true_distance'], pos_data['mean_rssi'],
                          alpha=0.6, label=pos, s=50)

        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Mean RSSI (dBm)', fontsize=11)
        ax.set_title('LoS Propagation', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # nLoS
        ax = axes[1]
        nlos_data = trajectories_df[trajectories_df['orientation'] == 'nlos']
        for pos in self.positions:
            pos_data = nlos_data[nlos_data['position'] == pos]
            if not pos_data.empty:
                ax.scatter(pos_data['true_distance'], pos_data['mean_rssi'],
                          alpha=0.6, label=pos, s=50)

        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Mean RSSI (dBm)', fontsize=11)
        ax.set_title('nLoS Propagation', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'rssi_propagation.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'rssi_propagation.pdf', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved RSSI propagation plots")

    def generate_report(self, trajectories_df, accuracy_analysis, output_dir):
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("B7: TRAJECTORY RECONSTRUCTION ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Overall Accuracy
        report.append("OVERALL TRACKING ACCURACY")
        report.append("-" * 80)
        report.append(f"Total trajectory points: {len(trajectories_df)}")
        report.append(f"Mean absolute error: {accuracy_analysis['overall']['mean_abs_error']:.2f} m")
        report.append(f"Root mean square error: {accuracy_analysis['overall']['rmse']:.2f} m")
        report.append(f"Mean relative error: {accuracy_analysis['overall']['mean_relative_error']:.1f}%")
        report.append(f"Mean bias (signed error): {accuracy_analysis['overall']['mean_error']:.2f} m")
        report.append("")

        # Accuracy by Distance
        report.append("ACCURACY BY DISTANCE")
        report.append("-" * 80)
        for dist in sorted(accuracy_analysis['by_distance'].keys()):
            stats = accuracy_analysis['by_distance'][dist]
            report.append(f"{dist}m:")
            report.append(f"  MAE: {stats['mae']:.2f} m")
            report.append(f"  Std: {stats['std']:.2f} m")
            report.append(f"  Relative error: {stats['relative_error']:.1f}%")
        report.append("")

        # Accuracy by Orientation
        report.append("ACCURACY BY ORIENTATION")
        report.append("-" * 80)
        for orient in sorted(accuracy_analysis['by_orientation'].keys()):
            stats = accuracy_analysis['by_orientation'][orient]
            report.append(f"{orient.upper()}:")
            report.append(f"  MAE: {stats['mae']:.2f} m")
            report.append(f"  Std: {stats['std']:.2f} m")
            report.append(f"  Relative error: {stats['relative_error']:.1f}%")
        report.append("")

        # Accuracy by Position
        report.append("ACCURACY BY POSITION")
        report.append("-" * 80)
        for pos in sorted(accuracy_analysis['by_position'].keys()):
            stats = accuracy_analysis['by_position'][pos]
            report.append(f"{pos}:")
            report.append(f"  MAE: {stats['mae']:.2f} m")
            report.append(f"  Std: {stats['std']:.2f} m")
            report.append(f"  Relative error: {stats['relative_error']:.1f}%")
        report.append("")

        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 80)

        # Determine best/worst distance
        best_dist = min(accuracy_analysis['by_distance'].items(),
                       key=lambda x: x[1]['mae'])
        worst_dist = max(accuracy_analysis['by_distance'].items(),
                        key=lambda x: x[1]['mae'])

        report.append(f"1. Best distance accuracy: {best_dist[0]}m (MAE: {best_dist[1]['mae']:.2f}m)")
        report.append(f"2. Worst distance accuracy: {worst_dist[0]}m (MAE: {worst_dist[1]['mae']:.2f}m)")

        # LoS vs nLoS
        if 'los' in accuracy_analysis['by_orientation'] and 'nlos' in accuracy_analysis['by_orientation']:
            los_mae = accuracy_analysis['by_orientation']['los']['mae']
            nlos_mae = accuracy_analysis['by_orientation']['nlos']['mae']
            report.append(f"3. LoS vs nLoS: {los_mae:.2f}m vs {nlos_mae:.2f}m MAE")
        elif 'los' in accuracy_analysis['by_orientation']:
            los_mae = accuracy_analysis['by_orientation']['los']['mae']
            report.append(f"3. LoS only: {los_mae:.2f}m MAE")
        elif 'nlos' in accuracy_analysis['by_orientation']:
            nlos_mae = accuracy_analysis['by_orientation']['nlos']['mae']
            report.append(f"3. nLoS only: {nlos_mae:.2f}m MAE")
        report.append(f"4. Overall relative error: {accuracy_analysis['overall']['mean_relative_error']:.1f}%")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("• Log-distance path loss model shows limited accuracy for trajectories")
        report.append("• Consider ML-based distance estimation (see B5) for better accuracy")
        report.append("• Combine RSSI with other sensors (IMU, GPS) for robust tracking")
        report.append("• Use Kalman filtering to smooth trajectory estimates")
        report.append("• Define tracking zones rather than precise positions")
        report.append("")

        # Limitations
        report.append("LIMITATIONS")
        report.append("-" * 80)
        report.append("• Static milestone measurements (not continuous walking trajectories)")
        report.append("• No GPS ground truth for continuous path validation")
        report.append("• Path loss model (from A1) not optimized for distance estimation")
        report.append("• Multipath interference creates large errors")
        report.append("• Body orientation changes affect signal strength")
        report.append("")

        report.append("=" * 80)

        # Write report
        report_text = "\n".join(report)
        with open(output_dir / 'trajectory_reconstruction_report.txt', 'w') as f:
            f.write(report_text)

        print(f"✓ Generated analysis report")
        return report_text

    def run_analysis(self):
        """Execute complete trajectory reconstruction analysis"""
        output_dir = Path('/home/user/BLE-Pedestrians/analysis/B7_trajectory_reconstruction/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("B7: TRAJECTORY RECONSTRUCTION ANALYSIS")
        print("=" * 80)

        # Load data
        print("\n1. Loading data...")
        los_data = self.load_data('los')
        nlos_data = self.load_data('nlos')
        all_data = pd.concat([los_data, nlos_data], ignore_index=True)
        print(f"   Loaded {len(all_data)} RSSI measurements")

        # Reconstruct trajectories
        print("\n2. Reconstructing trajectories...")
        trajectories_df = self.reconstruct_trajectories(all_data)
        trajectories_df.to_csv(output_dir / 'trajectories.csv', index=False)
        print(f"   Reconstructed {len(trajectories_df)} trajectory points")

        # Analyze accuracy
        print("\n3. Analyzing tracking accuracy...")
        accuracy_analysis = self.analyze_tracking_accuracy(trajectories_df)

        # Generate visualizations
        print("\n4. Creating visualizations...")
        self.plot_trajectory_reconstruction(trajectories_df, output_dir)
        self.plot_rssi_vs_distance(trajectories_df, output_dir)

        # Generate report
        print("\n5. Generating report...")
        report = self.generate_report(trajectories_df, accuracy_analysis, output_dir)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - trajectories.csv")
        print("  - trajectory_reconstruction.png/pdf")
        print("  - rssi_propagation.png/pdf")
        print("  - trajectory_reconstruction_report.txt")

        return trajectories_df, accuracy_analysis

if __name__ == '__main__':
    analyzer = TrajectoryReconstructionAnalyzer('/home/user/BLE-Pedestrians/dataset')
    trajectories_df, accuracy_analysis = analyzer.run_analysis()
