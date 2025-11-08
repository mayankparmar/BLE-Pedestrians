#!/usr/bin/env python3
"""
B6: Movement Detection & Direction Analysis

Distinguishes walking from stationary patterns and detects direction of movement
using RSSI temporal characteristics and GPS validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class MovementDetectionAnalyzer:
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

        # File prefix mapping for nLoS
        self.file_prefix_nlos = {
            'start': 'so',
            'mid_facing': 'mfo',
            'center': 'co',
            'mid_away': 'mao',
            'end': 'eo'
        }

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

    def detect_movement_from_gps(self, gps_df, speed_threshold=0.5):
        """Detect movement periods from GPS data (speed in m/s)"""
        # Calculate speed from consecutive GPS points
        if 'latitude' in gps_df.columns and 'longitude' in gps_df.columns:
            lat = gps_df['latitude'].values
            lon = gps_df['longitude'].values
            time_sec = (gps_df['timestamp'] - gps_df['timestamp'].iloc[0]).dt.total_seconds().values

            # Calculate distance between consecutive points
            lat_diff = np.diff(lat)
            lon_diff = np.diff(lon)
            # Approximate distance (meters) for small changes
            dist_m = np.sqrt((lat_diff * 111000)**2 + (lon_diff * 111000 * np.cos(np.radians(lat[:-1])))**2)

            time_diff = np.diff(time_sec)
            time_diff[time_diff == 0] = 0.001  # Avoid division by zero

            speed = dist_m / time_diff

            # Prepend zero for first point
            speed = np.concatenate([[0], speed])

            gps_df['speed'] = speed
            gps_df['moving'] = speed > speed_threshold
        else:
            gps_df['speed'] = 0
            gps_df['moving'] = False

        return gps_df

    def detect_movement_from_rssi(self, df, window_size=10):
        """Detect movement from RSSI temporal patterns"""
        movement_features = []

        for (dist, pos, orient, run), group in df.groupby(['distance', 'position', 'orientation', 'run']):
            if len(group) < window_size:
                continue

            rssi = group['rssi'].values
            time_sec = (group['timestamp'] - group['timestamp'].iloc[0]).dt.total_seconds().values

            # Calculate RSSI rate of change
            rssi_diff = np.diff(rssi)
            time_diff = np.diff(time_sec)
            time_diff[time_diff == 0] = 0.001
            rssi_rate = rssi_diff / time_diff

            # Calculate rolling statistics
            rssi_rolling_std = pd.Series(rssi).rolling(window=window_size, center=True).std().values
            rssi_rolling_range = pd.Series(rssi).rolling(window=window_size, center=True).apply(lambda x: x.max() - x.min()).values

            # Movement indicators
            rssi_variability = np.nanstd(rssi_rate)
            rssi_trend = np.polyfit(time_sec, rssi, 1)[0] if len(time_sec) > 1 else 0

            # From GPS
            if 'moving' in group.columns:
                gps_moving = group['moving'].any()
            else:
                gps_moving = None

            movement_features.append({
                'distance': dist,
                'position': pos,
                'orientation': orient,
                'run': run,
                'rssi_variability': rssi_variability,
                'rssi_trend': rssi_trend,
                'rssi_std': np.std(rssi),
                'rssi_range': np.max(rssi) - np.min(rssi),
                'mean_rssi_rate': np.nanmean(np.abs(rssi_rate)),
                'gps_moving': gps_moving,
                'duration_sec': time_sec[-1] - time_sec[0] if len(time_sec) > 1 else 0
            })

        return pd.DataFrame(movement_features)

    def analyze_direction(self, df):
        """Analyze approach vs departure from RSSI trend"""
        direction_analysis = []

        for (dist, pos, orient, run), group in df.groupby(['distance', 'position', 'orientation', 'run']):
            if len(group) < 5:
                continue

            rssi = group['rssi'].values
            time_sec = (group['timestamp'] - group['timestamp'].iloc[0]).dt.total_seconds().values

            # Linear fit to determine trend
            if len(time_sec) > 1:
                slope, intercept = np.polyfit(time_sec, rssi, 1)

                # Positive slope = approaching (signal getting stronger)
                # Negative slope = departing (signal getting weaker)
                direction = 'approaching' if slope > 0 else 'departing'

                # Calculate correlation to assess trend strength
                correlation = np.corrcoef(time_sec, rssi)[0, 1]

                direction_analysis.append({
                    'distance': dist,
                    'position': pos,
                    'orientation': orient,
                    'run': run,
                    'rssi_slope': slope,
                    'direction': direction,
                    'trend_strength': abs(correlation),
                    'mean_rssi': np.mean(rssi)
                })

        return pd.DataFrame(direction_analysis)

    def compare_static_vs_walking(self, df):
        """Compare signal characteristics during static vs walking"""
        # Use position as proxy: start/end are more dynamic, center is static milestone
        static_positions = ['center']
        dynamic_positions = ['start', 'mid_facing', 'mid_away', 'end']

        static_data = df[df['position'].isin(static_positions)]
        dynamic_data = df[df['position'].isin(dynamic_positions)]

        comparison = {
            'static': {
                'mean_std': static_data['rssi_std'].mean() if 'rssi_std' in static_data.columns else np.nan,
                'mean_variability': static_data['rssi_variability'].mean() if 'rssi_variability' in static_data.columns else np.nan,
                'mean_range': static_data['rssi_range'].mean() if 'rssi_range' in static_data.columns else np.nan
            },
            'dynamic': {
                'mean_std': dynamic_data['rssi_std'].mean() if 'rssi_std' in dynamic_data.columns else np.nan,
                'mean_variability': dynamic_data['rssi_variability'].mean() if 'rssi_variability' in dynamic_data.columns else np.nan,
                'mean_range': dynamic_data['rssi_range'].mean() if 'rssi_range' in dynamic_data.columns else np.nan
            }
        }

        return comparison

    def plot_movement_analysis(self, movement_features, direction_df, output_dir):
        """Create comprehensive movement analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('B6: Movement Detection & Direction Analysis', fontsize=16, fontweight='bold')

        # 1. RSSI Variability by Distance
        ax = axes[0, 0]
        for orient in ['los', 'nlos']:
            data = movement_features[movement_features['orientation'] == orient]
            dist_var = data.groupby('distance')['rssi_variability'].mean()
            ax.plot(dist_var.index, dist_var.values, marker='o', label=f'{orient.upper()}', linewidth=2)
        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('RSSI Variability (dB/s)', fontsize=11)
        ax.set_title('RSSI Variability vs Distance', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Movement Detection: RSSI Range
        ax = axes[0, 1]
        for orient in ['los', 'nlos']:
            data = movement_features[movement_features['orientation'] == orient]
            if not data.empty:
                ax.scatter(data['distance'], data['rssi_range'],
                          alpha=0.6, label=f'{orient.upper()}', s=50)
        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('RSSI Range (dB)', fontsize=11)
        ax.set_title('Signal Range Variability', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Direction Detection: Slope Distribution
        ax = axes[0, 2]
        if 'rssi_slope' in direction_df.columns:
            slopes = direction_df['rssi_slope'].values
            # Clip extreme values for better visualization
            slopes_clipped = np.clip(slopes, -5, 5)
            ax.hist(slopes_clipped, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Static (slope=0)')
            ax.set_xlabel('RSSI Slope (dB/s)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Direction Detection: RSSI Trend', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Direction by Distance
        ax = axes[1, 0]
        if not direction_df.empty:
            approach_data = direction_df[direction_df['direction'] == 'approaching']
            depart_data = direction_df[direction_df['direction'] == 'departing']

            distances = sorted(direction_df['distance'].unique())
            approach_counts = [len(approach_data[approach_data['distance'] == d]) for d in distances]
            depart_counts = [len(depart_data[depart_data['distance'] == d]) for d in distances]

            x = np.arange(len(distances))
            width = 0.35
            ax.bar(x - width/2, approach_counts, width, label='Approaching', alpha=0.8)
            ax.bar(x + width/2, depart_counts, width, label='Departing', alpha=0.8)
            ax.set_xlabel('Distance (m)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Direction Classification by Distance', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(distances)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # 5. Trend Strength Analysis
        ax = axes[1, 1]
        if 'trend_strength' in direction_df.columns:
            for orient in ['los', 'nlos']:
                data = direction_df[direction_df['orientation'] == orient]
                if not data.empty:
                    ax.scatter(data['distance'], data['trend_strength'],
                              alpha=0.6, label=f'{orient.upper()}', s=50)
            ax.set_xlabel('Distance (m)', fontsize=11)
            ax.set_ylabel('Trend Strength (|correlation|)', fontsize=11)
            ax.set_title('Movement Trend Reliability', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        # 6. RSSI Rate of Change
        ax = axes[1, 2]
        if 'mean_rssi_rate' in movement_features.columns:
            for orient in ['los', 'nlos']:
                data = movement_features[movement_features['orientation'] == orient]
                dist_rate = data.groupby('distance')['mean_rssi_rate'].mean()
                ax.plot(dist_rate.index, dist_rate.values, marker='o',
                       label=f'{orient.upper()}', linewidth=2)
            ax.set_xlabel('Distance (m)', fontsize=11)
            ax.set_ylabel('Mean |RSSI Rate| (dB/s)', fontsize=11)
            ax.set_title('Signal Change Rate', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'movement_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'movement_detection_analysis.pdf', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved movement analysis plots")

    def plot_temporal_patterns(self, df, output_dir):
        """Plot example temporal RSSI patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RSSI Temporal Patterns: Movement Examples', fontsize=16, fontweight='bold')

        # Select representative examples
        examples = [
            {'dist': 3, 'pos': 'start', 'orient': 'los'},
            {'dist': 5, 'pos': 'center', 'orient': 'los'},
            {'dist': 7, 'pos': 'end', 'orient': 'los'},
            {'dist': 9, 'pos': 'mid_facing', 'orient': 'nlos'}
        ]

        for idx, example in enumerate(examples):
            ax = axes[idx // 2, idx % 2]

            subset = df[
                (df['distance'] == example['dist']) &
                (df['position'] == example['pos']) &
                (df['orientation'] == example['orient']) &
                (df['run'] == 1)
            ]

            if not subset.empty:
                time_sec = (subset['timestamp'] - subset['timestamp'].iloc[0]).dt.total_seconds().values
                rssi = subset['rssi'].values

                # Plot raw RSSI
                ax.plot(time_sec, rssi, 'o-', alpha=0.6, markersize=3, label='Raw RSSI')

                # Fit trend line
                if len(time_sec) > 1:
                    slope, intercept = np.polyfit(time_sec, rssi, 1)
                    trend_line = slope * time_sec + intercept
                    ax.plot(time_sec, trend_line, 'r--', linewidth=2,
                           label=f'Trend (slope={slope:.2f} dB/s)')

                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('RSSI (dBm)', fontsize=11)
                ax.set_title(f"{example['dist']}m {example['pos']} ({example['orient'].upper()})",
                           fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'temporal_patterns.pdf', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved temporal pattern plots")

    def generate_report(self, movement_features, direction_df, comparison, output_dir):
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("B6: MOVEMENT DETECTION & DIRECTION ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Movement Detection Summary
        report.append("MOVEMENT DETECTION FROM RSSI PATTERNS")
        report.append("-" * 80)
        report.append(f"Total measurement sessions: {len(movement_features)}")
        report.append(f"Average RSSI variability: {movement_features['rssi_variability'].mean():.3f} dB/s")
        report.append(f"Average RSSI std deviation: {movement_features['rssi_std'].mean():.2f} dB")
        report.append(f"Average RSSI range: {movement_features['rssi_range'].mean():.2f} dB")
        report.append("")

        # Direction Detection
        report.append("DIRECTION DETECTION")
        report.append("-" * 80)
        if not direction_df.empty:
            approaching = len(direction_df[direction_df['direction'] == 'approaching'])
            departing = len(direction_df[direction_df['direction'] == 'departing'])
            total = len(direction_df)

            report.append(f"Approaching detections: {approaching} ({approaching/total*100:.1f}%)")
            report.append(f"Departing detections: {departing} ({departing/total*100:.1f}%)")
            report.append(f"Average trend strength: {direction_df['trend_strength'].mean():.3f}")
            report.append(f"Average RSSI slope: {direction_df['rssi_slope'].mean():.4f} dB/s")
        report.append("")

        # Static vs Dynamic Comparison
        report.append("STATIC VS DYNAMIC SIGNAL CHARACTERISTICS")
        report.append("-" * 80)
        report.append(f"Static positions (center):")
        report.append(f"  Mean std: {comparison['static']['mean_std']:.2f} dB")
        report.append(f"  Mean variability: {comparison['static']['mean_variability']:.3f} dB/s")
        report.append(f"  Mean range: {comparison['static']['mean_range']:.2f} dB")
        report.append("")
        report.append(f"Dynamic positions (start/end/mid):")
        report.append(f"  Mean std: {comparison['dynamic']['mean_std']:.2f} dB")
        report.append(f"  Mean variability: {comparison['dynamic']['mean_variability']:.3f} dB/s")
        report.append(f"  Mean range: {comparison['dynamic']['mean_range']:.2f} dB")
        report.append("")

        # Distance-specific analysis
        report.append("MOVEMENT CHARACTERISTICS BY DISTANCE")
        report.append("-" * 80)
        for dist in sorted(movement_features['distance'].unique()):
            dist_data = movement_features[movement_features['distance'] == dist]
            report.append(f"{dist}m:")
            report.append(f"  RSSI variability: {dist_data['rssi_variability'].mean():.3f} dB/s")
            report.append(f"  RSSI std: {dist_data['rssi_std'].mean():.2f} dB")
            report.append(f"  RSSI range: {dist_data['rssi_range'].mean():.2f} dB")
        report.append("")

        # Orientation comparison
        report.append("MOVEMENT DETECTION: LoS vs nLoS")
        report.append("-" * 80)
        for orient in ['los', 'nlos']:
            orient_data = movement_features[movement_features['orientation'] == orient]
            report.append(f"{orient.upper()}:")
            report.append(f"  RSSI variability: {orient_data['rssi_variability'].mean():.3f} dB/s")
            report.append(f"  RSSI std: {orient_data['rssi_std'].mean():.2f} dB")
            report.append(f"  Mean rate of change: {orient_data['mean_rssi_rate'].mean():.3f} dB/s")
        report.append("")

        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 80)
        report.append("1. Movement creates temporal RSSI patterns (rate of change)")
        report.append("2. Direction can be inferred from RSSI trend (positive=approach)")
        report.append("3. Static positions show lower variability than dynamic positions")
        report.append(f"4. Trend strength: {direction_df['trend_strength'].mean():.2f} correlation on average")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("• Use RSSI rate of change for movement detection")
        report.append("• Use RSSI slope sign for direction detection (+ = approaching)")
        report.append("• Apply smoothing (moving average) to reduce noise")
        report.append("• Combine with position tracking for trajectory reconstruction")
        report.append("• Account for multipath-induced temporal variations")
        report.append("")

        report.append("=" * 80)

        # Write report
        report_text = "\n".join(report)
        with open(output_dir / 'movement_detection_report.txt', 'w') as f:
            f.write(report_text)

        print(f"✓ Generated analysis report")
        return report_text

    def run_analysis(self):
        """Execute complete movement detection analysis"""
        output_dir = Path('/home/user/BLE-Pedestrians/analysis/B6_movement_detection/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("B6: MOVEMENT DETECTION & DIRECTION ANALYSIS")
        print("=" * 80)

        # Load data
        print("\n1. Loading data...")
        los_data = self.load_data('los')
        nlos_data = self.load_data('nlos')
        all_data = pd.concat([los_data, nlos_data], ignore_index=True)
        print(f"   Loaded {len(all_data)} RSSI measurements")

        # Detect movement from GPS
        print("\n2. Detecting movement from GPS...")
        all_data = self.detect_movement_from_gps(all_data)

        # Analyze RSSI temporal patterns
        print("\n3. Analyzing RSSI temporal patterns...")
        movement_features = self.detect_movement_from_rssi(all_data)
        movement_features.to_csv(output_dir / 'movement_features.csv', index=False)
        print(f"   Extracted {len(movement_features)} movement feature sets")

        # Direction analysis
        print("\n4. Analyzing movement direction...")
        direction_df = self.analyze_direction(all_data)
        direction_df.to_csv(output_dir / 'direction_analysis.csv', index=False)
        print(f"   Analyzed {len(direction_df)} direction patterns")

        # Compare static vs walking
        print("\n5. Comparing static vs dynamic patterns...")
        comparison = self.compare_static_vs_walking(movement_features)

        # Generate visualizations
        print("\n6. Creating visualizations...")
        self.plot_movement_analysis(movement_features, direction_df, output_dir)
        self.plot_temporal_patterns(all_data, output_dir)

        # Generate report
        print("\n7. Generating report...")
        report = self.generate_report(movement_features, direction_df, comparison, output_dir)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - movement_features.csv")
        print("  - direction_analysis.csv")
        print("  - movement_detection_analysis.png/pdf")
        print("  - temporal_patterns.png/pdf")
        print("  - movement_detection_report.txt")

        return movement_features, direction_df

if __name__ == '__main__':
    analyzer = MovementDetectionAnalyzer('/home/user/BLE-Pedestrians/dataset')
    movement_features, direction_df = analyzer.run_analysis()
