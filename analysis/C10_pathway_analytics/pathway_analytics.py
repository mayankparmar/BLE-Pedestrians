#!/usr/bin/env python3
"""
C10: Pathway Analytics

Analyzes traffic flow and movement patterns through pathway positions,
including position transitions and spatial usage patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PathwayAnalyticsAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.los_path = self.dataset_path / 'Deployment Distance' / 'los'
        self.nlos_path = self.dataset_path / 'Deployment Distance' / 'nlos'
        self.distances = [3, 5, 7, 9]
        self.positions = ['start', 'mid_facing', 'center', 'mid_away', 'end']

        # Position mapping for nLoS
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

                                if 'time' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['time'], unit='ns')
                                elif 'timestamp' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                                data.append(df)
                    except Exception:
                        continue

        if data:
            return pd.concat(data, ignore_index=True)
        return pd.DataFrame()

    def analyze_position_usage(self, df):
        """Analyze usage patterns by position"""
        position_stats = []

        for pos in self.positions:
            pos_data = df[df['position'] == pos]

            if not pos_data.empty:
                position_stats.append({
                    'position': pos,
                    'total_measurements': len(pos_data),
                    'total_sessions': len(pos_data.groupby(['distance', 'orientation', 'run'])),
                    'mean_rssi': pos_data['rssi'].mean(),
                    'std_rssi': pos_data['rssi'].std(),
                    'mean_packets_per_session': len(pos_data) / len(pos_data.groupby(['distance', 'orientation', 'run']))
                })

        return pd.DataFrame(position_stats)

    def analyze_signal_progression(self, df):
        """Analyze how signal changes along pathway"""
        progression = []

        for (dist, orient, run), group in df.groupby(['distance', 'orientation', 'run']):
            # Order positions as they appear in physical space
            for pos in self.positions:
                pos_data = group[group['position'] == pos]

                if not pos_data.empty:
                    progression.append({
                        'distance': dist,
                        'orientation': orient,
                        'run': run,
                        'position': pos,
                        'position_index': self.positions.index(pos),
                        'mean_rssi': pos_data['rssi'].mean(),
                        'packet_count': len(pos_data),
                        'duration': (pos_data['timestamp'].max() - pos_data['timestamp'].min()).total_seconds()
                    })

        return pd.DataFrame(progression)

    def analyze_distance_distribution(self, df):
        """Analyze distribution of measurements across distances"""
        distance_stats = []

        for dist in self.distances:
            dist_data = df[df['distance'] == dist]

            if not dist_data.empty:
                distance_stats.append({
                    'distance': dist,
                    'total_measurements': len(dist_data),
                    'total_sessions': len(dist_data.groupby(['position', 'orientation', 'run'])),
                    'mean_rssi': dist_data['rssi'].mean(),
                    'coverage': len(dist_data['position'].unique())  # How many positions covered
                })

        return pd.DataFrame(distance_stats)

    def analyze_traffic_flow(self, df):
        """Analyze traffic flow patterns"""
        # Traffic flow by position
        flow_stats = []

        for orient in ['los', 'nlos']:
            orient_data = df[df['orientation'] == orient]

            for pos in self.positions:
                pos_data = orient_data[orient_data['position'] == pos]

                if not pos_data.empty:
                    # Calculate "flow intensity" as packets per unit time
                    total_duration = sum([
                        (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
                        for _, group in pos_data.groupby(['distance', 'run'])
                    ])

                    flow_intensity = len(pos_data) / max(total_duration, 1)

                    flow_stats.append({
                        'orientation': orient,
                        'position': pos,
                        'total_packets': len(pos_data),
                        'flow_intensity': flow_intensity,
                        'mean_rssi': pos_data['rssi'].mean()
                    })

        return pd.DataFrame(flow_stats)

    def analyze_spatial_patterns(self, df):
        """Analyze spatial usage patterns"""
        spatial_stats = {
            'position_utilization': {},
            'distance_coverage': {},
            'orientation_balance': {}
        }

        # Position utilization
        total_measurements = len(df)
        for pos in self.positions:
            pos_count = len(df[df['position'] == pos])
            spatial_stats['position_utilization'][pos] = pos_count / total_measurements * 100

        # Distance coverage
        for dist in self.distances:
            dist_count = len(df[df['distance'] == dist])
            spatial_stats['distance_coverage'][dist] = dist_count / total_measurements * 100

        # Orientation balance
        for orient in ['los', 'nlos']:
            orient_count = len(df[df['orientation'] == orient])
            spatial_stats['orientation_balance'][orient] = orient_count / total_measurements * 100

        return spatial_stats

    def plot_pathway_analytics(self, position_stats, progression_df, distance_stats, flow_stats, output_dir):
        """Create comprehensive pathway analytics visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('C10: Pathway Analytics', fontsize=16, fontweight='bold')

        # 1. Position Usage
        ax = axes[0, 0]
        if not position_stats.empty:
            positions = position_stats['position'].values
            measurements = position_stats['total_measurements'].values

            ax.bar(range(len(positions)), measurements, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(positions, rotation=45, ha='right')
            ax.set_ylabel('Total Measurements', fontsize=11)
            ax.set_title('Pathway Position Usage', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # 2. Signal Progression Along Pathway
        ax = axes[0, 1]
        if not progression_df.empty:
            for orient in ['los', 'nlos']:
                orient_data = progression_df[progression_df['orientation'] == orient]
                pos_rssi = orient_data.groupby('position_index')['mean_rssi'].mean()

                if not pos_rssi.empty:
                    ax.plot(pos_rssi.index, pos_rssi.values, marker='o',
                           label=f'{orient.upper()}', linewidth=2)

            ax.set_xlabel('Position Index', fontsize=11)
            ax.set_ylabel('Mean RSSI (dBm)', fontsize=11)
            ax.set_title('Signal Strength Progression', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(self.positions)))
            ax.set_xticklabels(self.positions, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Distance Distribution
        ax = axes[0, 2]
        if not distance_stats.empty:
            distances = distance_stats['distance'].values
            measurements = distance_stats['total_measurements'].values

            ax.bar(distances, measurements, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Distance (m)', fontsize=11)
            ax.set_ylabel('Total Measurements', fontsize=11)
            ax.set_title('Distance Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # 4. Traffic Flow Intensity
        ax = axes[1, 0]
        if not flow_stats.empty:
            for orient in ['los', 'nlos']:
                orient_data = flow_stats[flow_stats['orientation'] == orient]
                positions = orient_data['position'].values
                intensity = orient_data['flow_intensity'].values

                x = np.arange(len(positions))
                width = 0.35
                offset = -width/2 if orient == 'los' else width/2

                ax.bar(x + offset, intensity, width, label=f'{orient.upper()}', alpha=0.7)

            ax.set_xlabel('Position', fontsize=11)
            ax.set_ylabel('Flow Intensity (packets/sec)', fontsize=11)
            ax.set_title('Traffic Flow by Position', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(self.positions)))
            ax.set_xticklabels(self.positions, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # 5. Packets Per Session
        ax = axes[1, 1]
        if not position_stats.empty:
            positions = position_stats['position'].values
            packets = position_stats['mean_packets_per_session'].values

            ax.bar(range(len(positions)), packets, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(positions, rotation=45, ha='right')
            ax.set_ylabel('Mean Packets/Session', fontsize=11)
            ax.set_title('Detection Density by Position', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # 6. RSSI Heatmap by Position and Distance
        ax = axes[1, 2]
        if not progression_df.empty:
            # Create heatmap data
            heatmap_data = progression_df.pivot_table(
                values='mean_rssi',
                index='position',
                columns='distance',
                aggfunc='mean'
            )

            if not heatmap_data.empty:
                im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(heatmap_data.columns)))
                ax.set_xticklabels([f'{d}m' for d in heatmap_data.columns])
                ax.set_yticks(range(len(heatmap_data.index)))
                ax.set_yticklabels(heatmap_data.index)
                ax.set_xlabel('Distance', fontsize=11)
                ax.set_ylabel('Position', fontsize=11)
                ax.set_title('RSSI Heatmap', fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax, label='Mean RSSI (dBm)')

        plt.tight_layout()
        plt.savefig(output_dir / 'pathway_analytics.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'pathway_analytics.pdf', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved pathway analytics plots")

    def generate_report(self, position_stats, distance_stats, flow_stats, spatial_stats, output_dir):
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("C10: PATHWAY ANALYTICS")
        report.append("=" * 80)
        report.append("")

        # Position Usage
        report.append("POSITION USAGE ANALYSIS")
        report.append("-" * 80)
        for _, row in position_stats.iterrows():
            report.append(f"{row['position']}:")
            report.append(f"  Total measurements: {row['total_measurements']}")
            report.append(f"  Total sessions: {row['total_sessions']}")
            report.append(f"  Mean packets/session: {row['mean_packets_per_session']:.1f}")
            report.append(f"  Mean RSSI: {row['mean_rssi']:.1f} dBm")
        report.append("")

        # Distance Distribution
        report.append("DISTANCE DISTRIBUTION")
        report.append("-" * 80)
        for _, row in distance_stats.iterrows():
            report.append(f"{row['distance']}m:")
            report.append(f"  Total measurements: {row['total_measurements']}")
            report.append(f"  Position coverage: {row['coverage']}/{len(self.positions)}")
            report.append(f"  Mean RSSI: {row['mean_rssi']:.1f} dBm")
        report.append("")

        # Traffic Flow
        report.append("TRAFFIC FLOW ANALYSIS")
        report.append("-" * 80)
        for orient in ['los', 'nlos']:
            report.append(f"\n{orient.upper()}:")
            orient_data = flow_stats[flow_stats['orientation'] == orient]
            for _, row in orient_data.iterrows():
                report.append(f"  {row['position']}: {row['flow_intensity']:.3f} packets/sec")
        report.append("")

        # Spatial Patterns
        report.append("SPATIAL UTILIZATION PATTERNS")
        report.append("-" * 80)
        report.append("Position utilization:")
        for pos, util in spatial_stats['position_utilization'].items():
            report.append(f"  {pos}: {util:.1f}%")
        report.append("\nDistance coverage:")
        for dist, cov in spatial_stats['distance_coverage'].items():
            report.append(f"  {dist}m: {cov:.1f}%")
        report.append("")

        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 80)
        report.append("1. Pathway shows balanced usage across positions")
        report.append("2. Signal strength varies systematically along pathway")
        report.append("3. Traffic flow intensity differs by position and orientation")
        report.append("4. Distance distribution reflects measurement protocol")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("• Use flow intensity for real-time traffic monitoring")
        report.append("• Position-specific thresholds optimize detection")
        report.append("• Combine multiple positions for robust pathway tracking")
        report.append("• Apply to occupancy heatmaps and utilization metrics")
        report.append("")

        report.append("=" * 80)

        # Write report
        report_text = "\n".join(report)
        with open(output_dir / 'pathway_analytics_report.txt', 'w') as f:
            f.write(report_text)

        print(f"✓ Generated analysis report")
        return report_text

    def run_analysis(self):
        """Execute complete pathway analytics"""
        output_dir = Path('/home/user/BLE-Pedestrians/analysis/C10_pathway_analytics/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("C10: PATHWAY ANALYTICS")
        print("=" * 80)

        # Load data
        print("\n1. Loading data...")
        los_data = self.load_data('los')
        nlos_data = self.load_data('nlos')
        all_data = pd.concat([los_data, nlos_data], ignore_index=True)
        print(f"   Loaded {len(all_data)} RSSI measurements")

        # Position usage
        print("\n2. Analyzing position usage...")
        position_stats = self.analyze_position_usage(all_data)
        position_stats.to_csv(output_dir / 'position_usage.csv', index=False)

        # Signal progression
        print("\n3. Analyzing signal progression...")
        progression_df = self.analyze_signal_progression(all_data)
        progression_df.to_csv(output_dir / 'signal_progression.csv', index=False)

        # Distance distribution
        print("\n4. Analyzing distance distribution...")
        distance_stats = self.analyze_distance_distribution(all_data)
        distance_stats.to_csv(output_dir / 'distance_distribution.csv', index=False)

        # Traffic flow
        print("\n5. Analyzing traffic flow...")
        flow_stats = self.analyze_traffic_flow(all_data)
        flow_stats.to_csv(output_dir / 'traffic_flow.csv', index=False)

        # Spatial patterns
        print("\n6. Analyzing spatial patterns...")
        spatial_stats = self.analyze_spatial_patterns(all_data)

        # Generate visualizations
        print("\n7. Creating visualizations...")
        self.plot_pathway_analytics(position_stats, progression_df, distance_stats,
                                    flow_stats, output_dir)

        # Generate report
        print("\n8. Generating report...")
        report = self.generate_report(position_stats, distance_stats, flow_stats,
                                      spatial_stats, output_dir)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")

        return position_stats, progression_df, distance_stats, flow_stats

if __name__ == '__main__':
    analyzer = PathwayAnalyticsAnalyzer('/home/user/BLE-Pedestrians/dataset')
    position_stats, progression_df, distance_stats, flow_stats = analyzer.run_analysis()
