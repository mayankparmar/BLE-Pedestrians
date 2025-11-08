#!/usr/bin/env python3
"""
D11: Signal Variability Analysis

Analyzes inter-run consistency and signal variability across repeated measurements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SignalVariabilityAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.los_path = self.dataset_path / 'Deployment Distance' / 'los'
        self.nlos_path = self.dataset_path / 'Deployment Distance' / 'nlos'
        self.distances = [3, 5, 7, 9]
        self.positions = ['start', 'mid_facing', 'center', 'mid_away', 'end']

        self.position_map_nlos = {
            'start': 'start',
            'mid_facing': 'mid facing',
            'center': 'centre',
            'mid_away': 'mid away',
            'end': 'end'
        }

    def load_data(self, orientation='los'):
        """Load RSSI data"""
        data = []
        base_path = self.los_path if orientation == 'los' else self.nlos_path

        for pos in self.positions:
            pos_dir = self.position_map_nlos.get(pos, pos) if orientation == 'nlos' else pos
            pos_path = base_path / pos_dir

            if not pos_path.exists():
                continue

            for run in range(1, 4):
                run_path = pos_path / f'run{run}'
                if not run_path.exists():
                    continue

                for dist in self.distances:
                    try:
                        data_file = run_path / f'{pos}_{dist}m_run{run}.csv' if orientation == 'los' else run_path / f'{pos_dir}_{dist}m_run{run}.csv'

                        if data_file.exists():
                            df = pd.read_csv(data_file)

                            if not df.empty and 'rssi' in df.columns:
                                df['distance'] = dist
                                df['position'] = pos
                                df['orientation'] = orientation
                                df['run'] = run
                                data.append(df)
                    except Exception:
                        continue

        if data:
            return pd.concat(data, ignore_index=True)
        return pd.DataFrame()

    def analyze_inter_run_variability(self, df):
        """Analyze variability between runs"""
        results = []

        for (dist, pos, orient), group in df.groupby(['distance', 'position', 'orientation']):
            # Get stats for each run
            run_stats = []
            for run in range(1, 4):
                run_data = group[group['run'] == run]
                if not run_data.empty:
                    run_stats.append({
                        'run': run,
                        'mean_rssi': run_data['rssi'].mean(),
                        'std_rssi': run_data['rssi'].std(),
                        'packet_count': len(run_data)
                    })

            if len(run_stats) >= 2:
                run_df = pd.DataFrame(run_stats)

                # Calculate inter-run variability
                results.append({
                    'distance': dist,
                    'position': pos,
                    'orientation': orient,
                    'inter_run_mean_std': run_df['mean_rssi'].std(),
                    'mean_rssi_across_runs': run_df['mean_rssi'].mean(),
                    'mean_within_run_std': run_df['std_rssi'].mean(),
                    'runs_available': len(run_stats),
                    'coefficient_of_variation': run_df['mean_rssi'].std() / abs(run_df['mean_rssi'].mean()) * 100 if run_df['mean_rssi'].mean() != 0 else 0
                })

        return pd.DataFrame(results)

    def analyze_temporal_variability(self, df):
        """Analyze temporal variability within measurements"""
        results = []

        for (dist, pos, orient, run), group in df.groupby(['distance', 'position', 'orientation', 'run']):
            if len(group) > 1:
                results.append({
                    'distance': dist,
                    'position': pos,
                    'orientation': orient,
                    'run': run,
                    'temporal_std': group['rssi'].std(),
                    'temporal_range': group['rssi'].max() - group['rssi'].min(),
                    'mean_rssi': group['rssi'].mean(),
                    'packet_count': len(group)
                })

        return pd.DataFrame(results)

    def analyze_consistency_metrics(self, inter_run_df):
        """Calculate consistency metrics"""
        metrics = {
            'overall': {
                'mean_inter_run_std': inter_run_df['inter_run_mean_std'].mean(),
                'mean_cv': inter_run_df['coefficient_of_variation'].mean()
            },
            'by_distance': {},
            'by_orientation': {}
        }

        # By distance
        for dist in self.distances:
            dist_data = inter_run_df[inter_run_df['distance'] == dist]
            if not dist_data.empty:
                metrics['by_distance'][dist] = {
                    'inter_run_std': dist_data['inter_run_mean_std'].mean(),
                    'cv': dist_data['coefficient_of_variation'].mean()
                }

        # By orientation
        for orient in ['los', 'nlos']:
            orient_data = inter_run_df[inter_run_df['orientation'] == orient]
            if not orient_data.empty:
                metrics['by_orientation'][orient] = {
                    'inter_run_std': orient_data['inter_run_mean_std'].mean(),
                    'cv': orient_data['coefficient_of_variation'].mean()
                }

        return metrics

    def plot_variability_analysis(self, inter_run_df, temporal_df, output_dir):
        """Create variability visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('D11: Signal Variability Analysis', fontsize=16, fontweight='bold')

        # 1. Inter-run variability by distance
        ax = axes[0, 0]
        for orient in ['los', 'nlos']:
            data = inter_run_df[inter_run_df['orientation'] == orient]
            dist_var = data.groupby('distance')['inter_run_mean_std'].mean()
            ax.plot(dist_var.index, dist_var.values, marker='o', label=orient.upper(), linewidth=2)

        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Inter-run Std Dev (dB)', fontsize=11)
        ax.set_title('Inter-Run Variability', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Coefficient of variation
        ax = axes[0, 1]
        for orient in ['los', 'nlos']:
            data = inter_run_df[inter_run_df['orientation'] == orient]
            dist_cv = data.groupby('distance')['coefficient_of_variation'].mean()
            ax.plot(dist_cv.index, dist_cv.values, marker='o', label=orient.upper(), linewidth=2)

        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
        ax.set_title('Signal Consistency (CV)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Inter-run vs within-run variability
        ax = axes[0, 2]
        ax.scatter(inter_run_df['mean_within_run_std'], inter_run_df['inter_run_mean_std'], alpha=0.5)
        max_val = max(inter_run_df['mean_within_run_std'].max(), inter_run_df['inter_run_mean_std'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Equal variability')
        ax.set_xlabel('Within-Run Std Dev (dB)', fontsize=11)
        ax.set_ylabel('Inter-Run Std Dev (dB)', fontsize=11)
        ax.set_title('Variability Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Temporal variability distribution
        ax = axes[1, 0]
        ax.hist(temporal_df['temporal_std'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(temporal_df['temporal_std'].mean(), color='red', linestyle='--',
                  label=f"Mean: {temporal_df['temporal_std'].mean():.2f} dB")
        ax.set_xlabel('Temporal Std Dev (dB)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Temporal Variability Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 5. Variability by position
        ax = axes[1, 1]
        pos_var = inter_run_df.groupby('position')['inter_run_mean_std'].mean()
        ax.bar(range(len(pos_var)), pos_var.values, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(pos_var)))
        ax.set_xticklabels(pos_var.index, rotation=45, ha='right')
        ax.set_ylabel('Inter-Run Std Dev (dB)', fontsize=11)
        ax.set_title('Variability by Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Signal range
        ax = axes[1, 2]
        for orient in ['los', 'nlos']:
            data = temporal_df[temporal_df['orientation'] == orient]
            dist_range = data.groupby('distance')['temporal_range'].mean()
            ax.plot(dist_range.index, dist_range.values, marker='o', label=orient.upper(), linewidth=2)

        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Mean Signal Range (dB)', fontsize=11)
        ax.set_title('Temporal Signal Range', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'signal_variability_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'signal_variability_analysis.pdf', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved variability analysis plots")

    def generate_report(self, inter_run_df, consistency_metrics, output_dir):
        """Generate report"""
        report = []
        report.append("=" * 80)
        report.append("D11: SIGNAL VARIABILITY ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Overall metrics
        report.append("OVERALL CONSISTENCY METRICS")
        report.append("-" * 80)
        report.append(f"Mean inter-run std dev: {consistency_metrics['overall']['mean_inter_run_std']:.2f} dB")
        report.append(f"Mean coefficient of variation: {consistency_metrics['overall']['mean_cv']:.2f}%")
        report.append("")

        # By distance
        report.append("VARIABILITY BY DISTANCE")
        report.append("-" * 80)
        for dist in sorted(consistency_metrics['by_distance'].keys()):
            stats = consistency_metrics['by_distance'][dist]
            report.append(f"{dist}m:")
            report.append(f"  Inter-run std: {stats['inter_run_std']:.2f} dB")
            report.append(f"  CV: {stats['cv']:.2f}%")
        report.append("")

        # By orientation
        report.append("VARIABILITY BY ORIENTATION")
        report.append("-" * 80)
        for orient in sorted(consistency_metrics['by_orientation'].keys()):
            stats = consistency_metrics['by_orientation'][orient]
            report.append(f"{orient.upper()}:")
            report.append(f"  Inter-run std: {stats['inter_run_std']:.2f} dB")
            report.append(f"  CV: {stats['cv']:.2f}%")
        report.append("")

        report.append("KEY INSIGHTS")
        report.append("-" * 80)
        report.append("1. Inter-run variability quantifies measurement consistency")
        report.append("2. Coefficient of variation normalizes variability across signal strengths")
        report.append("3. Both temporal and inter-run variability contribute to uncertainty")
        report.append("")

        report.append("=" * 80)

        with open(output_dir / 'signal_variability_report.txt', 'w') as f:
            f.write("\n".join(report))

        print(f"✓ Generated analysis report")

    def run_analysis(self):
        """Execute analysis"""
        output_dir = Path('/home/user/BLE-Pedestrians/analysis/D11_signal_variability/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("D11: SIGNAL VARIABILITY ANALYSIS")
        print("=" * 80)

        print("\n1. Loading data...")
        los_data = self.load_data('los')
        nlos_data = self.load_data('nlos')
        all_data = pd.concat([los_data, nlos_data], ignore_index=True)
        print(f"   Loaded {len(all_data)} measurements")

        print("\n2. Analyzing inter-run variability...")
        inter_run_df = self.analyze_inter_run_variability(all_data)
        inter_run_df.to_csv(output_dir / 'inter_run_variability.csv', index=False)

        print("\n3. Analyzing temporal variability...")
        temporal_df = self.analyze_temporal_variability(all_data)
        temporal_df.to_csv(output_dir / 'temporal_variability.csv', index=False)

        print("\n4. Calculating consistency metrics...")
        consistency_metrics = self.analyze_consistency_metrics(inter_run_df)

        print("\n5. Creating visualizations...")
        self.plot_variability_analysis(inter_run_df, temporal_df, output_dir)

        print("\n6. Generating report...")
        self.generate_report(inter_run_df, consistency_metrics, output_dir)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return inter_run_df, temporal_df

if __name__ == '__main__':
    analyzer = SignalVariabilityAnalyzer('/home/user/BLE-Pedestrians/dataset')
    inter_run_df, temporal_df = analyzer.run_analysis()
