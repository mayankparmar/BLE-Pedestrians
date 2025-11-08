"""
Environmental Multipath Analysis
=================================
This script compares controlled anechoic chamber measurements with real-world field data
to characterize multipath fading, environmental effects, and angular signal dependency.

Key Analyses:
1. Anechoic chamber angular pattern analysis (0° to 180°)
2. Field measurement variability vs. controlled environment
3. Multipath fading characterization
4. Environmental impact quantification
5. Signal stability comparison

Author: Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class MultipathAnalyzer:
    """Analyzes multipath effects by comparing anechoic vs field measurements"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.anechoic_angles = [0, 45, 90, 135, 180]  # degrees
        self.field_distances = [3, 5]  # meters
        self.field_configurations = ['fb', 'bt', 'bd']  # front-back, back-top, back-down

    def load_anechoic_data(self):
        """
        Load anechoic chamber data for all angles

        Returns:
            dict: {angle: DataFrame}
        """
        anechoic_data = {}
        anechoic_path = self.dataset_path / "Anechoic Chamber"

        for angle in self.anechoic_angles:
            file_path = anechoic_path / f"anechoic-{angle}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    anechoic_data[angle] = df
                    print(f"Loaded anechoic data for {angle}°: {len(df)} samples")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Warning: {file_path} not found")

        return anechoic_data

    def load_field_data(self):
        """
        Load field test data for all configurations

        Returns:
            dict: {config_distance: DataFrame}
        """
        field_data = {}

        for config in self.field_configurations:
            for distance in self.field_distances:
                for suffix in ['', '_occ']:  # with and without occlusion
                    key = f"{config}_{distance}m{suffix}"
                    file_path = self.dataset_path / f"{config}_{distance}-metres{suffix}.csv"

                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path)
                            field_data[key] = df
                            print(f"Loaded field data for {key}: {len(df)} samples")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")

        return field_data

    def analyze_anechoic_angular_pattern(self, anechoic_data):
        """
        Analyze angular dependency of RSSI in controlled environment

        Args:
            anechoic_data: Dict of anechoic chamber DataFrames

        Returns:
            DataFrame with angular statistics
        """
        results = []

        for angle, df in anechoic_data.items():
            if 'rssi' in df.columns:
                rssi_values = df['rssi'].dropna()

                results.append({
                    'angle': angle,
                    'mean_rssi': rssi_values.mean(),
                    'median_rssi': rssi_values.median(),
                    'std_rssi': rssi_values.std(),
                    'min_rssi': rssi_values.min(),
                    'max_rssi': rssi_values.max(),
                    'range_rssi': rssi_values.max() - rssi_values.min(),
                    'iqr_rssi': rssi_values.quantile(0.75) - rssi_values.quantile(0.25),
                    'count': len(rssi_values)
                })

        return pd.DataFrame(results)

    def analyze_field_variability(self, field_data):
        """
        Analyze signal variability in field measurements

        Args:
            field_data: Dict of field test DataFrames

        Returns:
            DataFrame with field statistics
        """
        results = []

        for config_key, df in field_data.items():
            if 'rssi' in df.columns:
                rssi_values = df['rssi'].dropna()
            elif 'mean' in df.columns:
                rssi_values = df['mean'].dropna()
            else:
                continue

            # Parse configuration
            parts = config_key.split('_')
            config_type = parts[0]
            distance = int(parts[1].replace('m', ''))
            occluded = 'occ' in config_key

            results.append({
                'configuration': config_type,
                'distance_m': distance,
                'occluded': occluded,
                'config_key': config_key,
                'mean_rssi': rssi_values.mean(),
                'median_rssi': rssi_values.median(),
                'std_rssi': rssi_values.std(),
                'min_rssi': rssi_values.min(),
                'max_rssi': rssi_values.max(),
                'range_rssi': rssi_values.max() - rssi_values.min(),
                'iqr_rssi': rssi_values.quantile(0.75) - rssi_values.quantile(0.25),
                'count': len(rssi_values)
            })

        return pd.DataFrame(results)

    def compare_stability(self, anechoic_stats, field_stats):
        """
        Compare signal stability between anechoic and field environments

        Args:
            anechoic_stats: Anechoic chamber statistics DataFrame
            field_stats: Field test statistics DataFrame

        Returns:
            dict with comparison metrics
        """
        # Average anechoic stability
        anechoic_avg_std = anechoic_stats['std_rssi'].mean()
        anechoic_avg_range = anechoic_stats['range_rssi'].mean()

        # Field stability (non-occluded only for fair comparison)
        field_clear = field_stats[~field_stats['occluded']]
        field_avg_std = field_clear['std_rssi'].mean()
        field_avg_range = field_clear['range_rssi'].mean()

        # Multipath factor (how much more variable is field vs anechoic)
        multipath_factor_std = field_avg_std / anechoic_avg_std if anechoic_avg_std > 0 else np.inf
        multipath_factor_range = field_avg_range / anechoic_avg_range if anechoic_avg_range > 0 else np.inf

        return {
            'anechoic_avg_std': anechoic_avg_std,
            'anechoic_avg_range': anechoic_avg_range,
            'field_avg_std': field_avg_std,
            'field_avg_range': field_avg_range,
            'multipath_factor_std': multipath_factor_std,
            'multipath_factor_range': multipath_factor_range,
            'std_increase_db': field_avg_std - anechoic_avg_std,
            'range_increase_db': field_avg_range - anechoic_avg_range
        }

    def calculate_fading_statistics(self, field_data):
        """
        Calculate fading statistics for field measurements

        Args:
            field_data: Dict of field test DataFrames

        Returns:
            DataFrame with fading characteristics
        """
        results = []

        for config_key, df in field_data.items():
            if 'rssi' in df.columns:
                rssi_values = df['rssi'].dropna().values
            elif 'mean' in df.columns:
                rssi_values = df['mean'].dropna().values
            else:
                continue

            if len(rssi_values) < 2:
                continue

            # Calculate temporal fading (RSSI variations over time)
            rssi_diff = np.diff(rssi_values)

            # Fade depth (difference from mean)
            fade_depth = rssi_values - np.mean(rssi_values)

            # Parse configuration
            parts = config_key.split('_')
            config_type = parts[0]
            distance = int(parts[1].replace('m', ''))
            occluded = 'occ' in config_key

            results.append({
                'configuration': config_type,
                'distance_m': distance,
                'occluded': occluded,
                'config_key': config_key,
                'fade_depth_std': np.std(fade_depth),
                'fade_depth_max': np.max(np.abs(fade_depth)),
                'temporal_variation_std': np.std(rssi_diff),
                'temporal_variation_max': np.max(np.abs(rssi_diff)),
                'fade_rate_per_sample': np.mean(np.abs(rssi_diff))
            })

        return pd.DataFrame(results)

    def plot_angular_pattern(self, anechoic_stats, output_dir):
        """
        Visualize angular signal pattern from anechoic chamber

        Args:
            anechoic_stats: Anechoic statistics DataFrame
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Sort by angle
        anechoic_stats = anechoic_stats.sort_values('angle')
        angles = anechoic_stats['angle'].values

        # 1. Mean RSSI vs Angle
        ax1 = axes[0, 0]
        ax1.plot(angles, anechoic_stats['mean_rssi'], 'o-', markersize=10,
                linewidth=2, color='blue', label='Mean RSSI')
        ax1.fill_between(angles,
                         anechoic_stats['mean_rssi'] - anechoic_stats['std_rssi'],
                         anechoic_stats['mean_rssi'] + anechoic_stats['std_rssi'],
                         alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax1.set_ylabel('RSSI (dBm)', fontsize=12)
        ax1.set_title('Anechoic Chamber: RSSI vs Rotation Angle', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xticks(angles)

        # 2. Signal Variability vs Angle
        ax2 = axes[0, 1]
        ax2.plot(angles, anechoic_stats['std_rssi'], 's-', markersize=10,
                linewidth=2, color='red', label='Std Deviation')
        ax2.plot(angles, anechoic_stats['range_rssi'], '^-', markersize=10,
                linewidth=2, color='orange', label='Range (Max-Min)')
        ax2.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax2.set_ylabel('Signal Variability (dB)', fontsize=12)
        ax2.set_title('Signal Variability vs Rotation Angle', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xticks(angles)

        # 3. Polar Plot of RSSI
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        # Convert angles to radians
        angles_rad = np.deg2rad(angles)
        # Normalize RSSI for polar plot (make positive)
        rssi_normalized = anechoic_stats['mean_rssi'] + abs(anechoic_stats['mean_rssi'].min()) + 5

        ax3.plot(angles_rad, rssi_normalized, 'o-', markersize=10, linewidth=2, color='blue')
        ax3.fill(angles_rad, rssi_normalized, alpha=0.3, color='blue')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_title('Polar Radiation Pattern\n(Anechoic Chamber)', fontsize=13, fontweight='bold', pad=20)
        ax3.set_ylim(0, max(rssi_normalized) * 1.1)

        # 4. Min/Max Range
        ax4 = axes[1, 1]
        x_pos = np.arange(len(angles))
        width = 0.35

        bars1 = ax4.bar(x_pos - width/2, anechoic_stats['min_rssi'], width,
                       label='Min RSSI', color='lightblue', edgecolor='blue', linewidth=1.5)
        bars2 = ax4.bar(x_pos + width/2, anechoic_stats['max_rssi'], width,
                       label='Max RSSI', color='lightcoral', edgecolor='red', linewidth=1.5)

        ax4.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax4.set_ylabel('RSSI (dBm)', fontsize=12)
        ax4.set_title('RSSI Range by Angle', fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(angles)
        ax4.legend(fontsize=10)
        ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'anechoic_angular_pattern.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'anechoic_angular_pattern.pdf', bbox_inches='tight')
        print(f"Saved angular pattern plots to {output_dir}")
        plt.close()

    def plot_environment_comparison(self, anechoic_stats, field_stats, comparison, output_dir):
        """
        Compare anechoic chamber vs field measurements

        Args:
            anechoic_stats: Anechoic statistics DataFrame
            field_stats: Field statistics DataFrame
            comparison: Comparison metrics dict
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. RSSI Distribution Comparison
        ax1 = axes[0, 0]

        # Anechoic chamber
        anechoic_mean = anechoic_stats['mean_rssi'].values
        anechoic_std = anechoic_stats['std_rssi'].values

        # Field measurements (separate by distance and occlusion)
        field_3m = field_stats[(field_stats['distance_m'] == 3) & (~field_stats['occluded'])]
        field_5m = field_stats[(field_stats['distance_m'] == 5) & (~field_stats['occluded'])]

        categories = ['Anechoic\n(3m)', 'Field 3m\n(Clear)', 'Field 5m\n(Clear)']
        means = [
            anechoic_stats['mean_rssi'].mean(),
            field_3m['mean_rssi'].mean() if not field_3m.empty else np.nan,
            field_5m['mean_rssi'].mean() if not field_5m.empty else np.nan
        ]
        stds = [
            anechoic_stats['std_rssi'].mean(),
            field_3m['std_rssi'].mean() if not field_3m.empty else np.nan,
            field_5m['std_rssi'].mean() if not field_5m.empty else np.nan
        ]

        x_pos = np.arange(len(categories))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=10,
                      color=['green', 'blue', 'orange'], alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        ax1.set_ylabel('Mean RSSI (dBm)', fontsize=12)
        ax1.set_title('RSSI Comparison: Controlled vs Field', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories)
        ax1.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if not np.isnan(mean):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., mean,
                        f'{mean:.1f}±{std:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. Signal Stability Comparison
        ax2 = axes[0, 1]

        stability_categories = ['Anechoic', 'Field (Clear)', 'Field (Occluded)']
        field_occluded = field_stats[field_stats['occluded']]

        std_devs = [
            anechoic_stats['std_rssi'].mean(),
            field_stats[~field_stats['occluded']]['std_rssi'].mean(),
            field_occluded['std_rssi'].mean() if not field_occluded.empty else np.nan
        ]
        ranges = [
            anechoic_stats['range_rssi'].mean(),
            field_stats[~field_stats['occluded']]['range_rssi'].mean(),
            field_occluded['range_rssi'].mean() if not field_occluded.empty else np.nan
        ]

        x_pos = np.arange(len(stability_categories))
        width = 0.35

        bars1 = ax2.bar(x_pos - width/2, std_devs, width, label='Std Dev',
                       color='skyblue', edgecolor='blue', linewidth=1.5)
        bars2 = ax2.bar(x_pos + width/2, ranges, width, label='Range',
                       color='lightcoral', edgecolor='red', linewidth=1.5)

        ax2.set_ylabel('Signal Variability (dB)', fontsize=12)
        ax2.set_title('Signal Stability: Anechoic vs Field\n(Lower is More Stable)',
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stability_categories, fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, axis='y', alpha=0.3)

        # 3. Multipath Factor
        ax3 = axes[1, 0]

        multipath_metrics = ['Std Dev Ratio', 'Range Ratio']
        multipath_values = [
            comparison['multipath_factor_std'],
            comparison['multipath_factor_range']
        ]

        bars = ax3.bar(multipath_metrics, multipath_values,
                      color=['purple', 'brown'], alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                   label='Baseline (Anechoic = 1.0)')
        ax3.set_ylabel('Multipath Factor (Field/Anechoic)', fontsize=12)
        ax3.set_title('Multipath Effect Quantification', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, multipath_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}x',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 4. Configuration Comparison
        ax4 = axes[1, 1]

        if not field_stats.empty:
            # Group by configuration and distance
            for config in field_stats['configuration'].unique():
                config_data = field_stats[(field_stats['configuration'] == config) &
                                         (~field_stats['occluded'])]
                if not config_data.empty:
                    distances = config_data['distance_m'].values
                    stds = config_data['std_rssi'].values
                    ax4.plot(distances, stds, 'o-', markersize=10, linewidth=2,
                            label=f'{config.upper()}', alpha=0.8)

            ax4.axhline(y=anechoic_stats['std_rssi'].mean(), color='green',
                       linestyle='--', linewidth=2, label='Anechoic Baseline')

            ax4.set_xlabel('Distance (m)', fontsize=12)
            ax4.set_ylabel('RSSI Std Dev (dB)', fontsize=12)
            ax4.set_title('Field Variability by Configuration', fontsize=13, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'environment_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'environment_comparison.pdf', bbox_inches='tight')
        print(f"Saved environment comparison plots to {output_dir}")
        plt.close()

    def plot_fading_analysis(self, fading_stats, output_dir):
        """
        Visualize fading characteristics

        Args:
            fading_stats: Fading statistics DataFrame
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Fade Depth by Configuration
        ax1 = axes[0, 0]

        clear_data = fading_stats[~fading_stats['occluded']]
        if not clear_data.empty:
            configs = clear_data['configuration'].unique()
            x_pos = np.arange(len(configs))

            fade_depths_3m = []
            fade_depths_5m = []

            for config in configs:
                data_3m = clear_data[(clear_data['configuration'] == config) &
                                    (clear_data['distance_m'] == 3)]
                data_5m = clear_data[(clear_data['configuration'] == config) &
                                    (clear_data['distance_m'] == 5)]

                fade_depths_3m.append(data_3m['fade_depth_std'].mean() if not data_3m.empty else 0)
                fade_depths_5m.append(data_5m['fade_depth_std'].mean() if not data_5m.empty else 0)

            width = 0.35
            ax1.bar(x_pos - width/2, fade_depths_3m, width, label='3m',
                   color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax1.bar(x_pos + width/2, fade_depths_5m, width, label='5m',
                   color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)

            ax1.set_ylabel('Fade Depth Std Dev (dB)', fontsize=12)
            ax1.set_title('Fading Depth by Configuration', fontsize=13, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([c.upper() for c in configs])
            ax1.legend(fontsize=10)
            ax1.grid(True, axis='y', alpha=0.3)

        # 2. Temporal Variation
        ax2 = axes[0, 1]

        if not clear_data.empty:
            for config in configs:
                config_data = clear_data[clear_data['configuration'] == config]
                distances = config_data['distance_m'].values
                temp_vars = config_data['temporal_variation_std'].values

                ax2.plot(distances, temp_vars, 'o-', markersize=10, linewidth=2,
                        label=f'{config.upper()}', alpha=0.8)

            ax2.set_xlabel('Distance (m)', fontsize=12)
            ax2.set_ylabel('Temporal Variation Std (dB)', fontsize=12)
            ax2.set_title('Signal Temporal Variability', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # 3. Occlusion Effect
        ax3 = axes[1, 0]

        if not fading_stats.empty:
            # Compare occluded vs non-occluded
            for distance in [3, 5]:
                clear = fading_stats[(fading_stats['distance_m'] == distance) &
                                    (~fading_stats['occluded'])]
                occluded = fading_stats[(fading_stats['distance_m'] == distance) &
                                       (fading_stats['occluded'])]

                if not clear.empty and not occluded.empty:
                    categories = [f'{d}m\nClear' for d in [distance]] + [f'{d}m\nOccluded' for d in [distance]]
                    values = [clear['fade_depth_std'].mean(), occluded['fade_depth_std'].mean()]

                    colors = ['green' if 'Clear' in cat else 'red' for cat in categories]
                    x_pos = np.arange(len(categories)) + (distance - 3) * 2.5

                    ax3.bar(x_pos, values, color=colors, alpha=0.7,
                           edgecolor='black', linewidth=1.5, width=0.8)

            ax3.set_ylabel('Fade Depth Std Dev (dB)', fontsize=12)
            ax3.set_title('Occlusion Effect on Fading', fontsize=13, fontweight='bold')
            ax3.grid(True, axis='y', alpha=0.3)

        # 4. Fade Rate
        ax4 = axes[1, 1]

        if not clear_data.empty:
            for config in configs:
                config_data = clear_data[clear_data['configuration'] == config]
                distances = config_data['distance_m'].values
                fade_rates = config_data['fade_rate_per_sample'].values

                ax4.plot(distances, fade_rates, 'o-', markersize=10, linewidth=2,
                        label=f'{config.upper()}', alpha=0.8)

            ax4.set_xlabel('Distance (m)', fontsize=12)
            ax4.set_ylabel('Fade Rate (dB/sample)', fontsize=12)
            ax4.set_title('Signal Fade Rate', fontsize=13, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'fading_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fading_analysis.pdf', bbox_inches='tight')
        print(f"Saved fading analysis plots to {output_dir}")
        plt.close()

    def generate_report(self, anechoic_stats, field_stats, comparison, fading_stats, output_dir):
        """
        Generate comprehensive analysis report

        Args:
            anechoic_stats: Anechoic statistics DataFrame
            field_stats: Field statistics DataFrame
            comparison: Comparison metrics dict
            fading_stats: Fading statistics DataFrame
            output_dir: Output directory
        """
        report = []
        report.append("=" * 80)
        report.append("ENVIRONMENTAL MULTIPATH ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("EXPERIMENTAL SETUP")
        report.append("-" * 80)
        report.append("Anechoic Chamber:")
        report.append(f"  - Rotation angles: {self.anechoic_angles} degrees")
        report.append(f"  - Fixed distance: 3 meters")
        report.append(f"  - Controlled environment (no multipath)")
        report.append("")
        report.append("Field Tests:")
        report.append(f"  - Configurations: {self.field_configurations}")
        report.append(f"  - Distances: {self.field_distances} meters")
        report.append(f"  - Real-world environment with multipath")
        report.append("")

        report.append("ANECHOIC CHAMBER RESULTS (Controlled Environment)")
        report.append("-" * 80)
        for _, row in anechoic_stats.iterrows():
            report.append(f"\nAngle {int(row['angle'])}°:")
            report.append(f"  Mean RSSI: {row['mean_rssi']:.2f} ± {row['std_rssi']:.2f} dBm")
            report.append(f"  Range: {row['min_rssi']:.1f} to {row['max_rssi']:.1f} dBm ({row['range_rssi']:.1f} dB)")
            report.append(f"  Samples: {int(row['count'])}")

        report.append(f"\nOverall Anechoic Stability:")
        report.append(f"  Average Std Dev: {anechoic_stats['std_rssi'].mean():.2f} dB")
        report.append(f"  Average Range: {anechoic_stats['range_rssi'].mean():.2f} dB")
        report.append("")

        report.append("FIELD MEASUREMENT RESULTS (Real-World Environment)")
        report.append("-" * 80)

        # Group by configuration
        for config in field_stats['configuration'].unique():
            config_data = field_stats[field_stats['configuration'] == config]
            report.append(f"\nConfiguration: {config.upper()}")

            for distance in sorted(config_data['distance_m'].unique()):
                dist_data = config_data[config_data['distance_m'] == distance]

                for occluded in [False, True]:
                    occ_data = dist_data[dist_data['occluded'] == occluded]
                    if occ_data.empty:
                        continue

                    occ_label = "Occluded" if occluded else "Clear"
                    report.append(f"  {distance}m ({occ_label}):")
                    report.append(f"    Mean RSSI: {occ_data['mean_rssi'].mean():.2f} ± {occ_data['std_rssi'].mean():.2f} dBm")
                    report.append(f"    Range: {occ_data['range_rssi'].mean():.1f} dB")

        report.append("")
        report.append("MULTIPATH EFFECT QUANTIFICATION")
        report.append("-" * 80)
        report.append(f"Anechoic Chamber Stability (Baseline):")
        report.append(f"  Std Dev: {comparison['anechoic_avg_std']:.2f} dB")
        report.append(f"  Range: {comparison['anechoic_avg_range']:.2f} dB")
        report.append("")
        report.append(f"Field Environment Stability (Clear conditions):")
        report.append(f"  Std Dev: {comparison['field_avg_std']:.2f} dB")
        report.append(f"  Range: {comparison['field_avg_range']:.2f} dB")
        report.append("")
        report.append(f"Multipath Impact:")
        report.append(f"  Std Dev Increase: {comparison['std_increase_db']:.2f} dB ({comparison['multipath_factor_std']:.2f}x)")
        report.append(f"  Range Increase: {comparison['range_increase_db']:.2f} dB ({comparison['multipath_factor_range']:.2f}x)")
        report.append("")

        report.append("INTERPRETATION")
        report.append("-" * 80)

        mp_factor = comparison['multipath_factor_std']
        if mp_factor < 1.5:
            report.append("Minimal multipath effect - field environment is relatively clean")
        elif mp_factor < 2.5:
            report.append("Moderate multipath effect - typical indoor/outdoor environment")
        else:
            report.append("Significant multipath effect - complex environment with reflections")

        report.append("")
        report.append(f"The field environment shows {mp_factor:.1f}x more signal variability")
        report.append(f"compared to the controlled anechoic chamber, indicating multipath")
        report.append(f"propagation from reflections, scattering, and environmental factors.")

        report.append("")
        report.append("=" * 80)

        # Save report
        report_text = "\n".join(report)
        with open(output_dir / 'multipath_analysis_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {output_dir / 'multipath_analysis_report.txt'}")


def main():
    """Main execution function"""

    # Setup paths
    base_path = Path("/home/user/BLE-Pedestrians")
    dataset_path = base_path / "dataset" / "Directionality"
    output_dir = base_path / "analysis" / "A3_environmental_multipath" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENVIRONMENTAL MULTIPATH ANALYSIS")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = MultipathAnalyzer(dataset_path)

    # Load data
    print("Loading anechoic chamber data...")
    anechoic_data = analyzer.load_anechoic_data()
    print()

    print("Loading field test data...")
    field_data = analyzer.load_field_data()
    print()

    # Analyze anechoic chamber
    print("Analyzing anechoic chamber angular patterns...")
    anechoic_stats = analyzer.analyze_anechoic_angular_pattern(anechoic_data)
    anechoic_stats.to_csv(output_dir / 'anechoic_statistics.csv', index=False)
    print()

    # Analyze field measurements
    print("Analyzing field measurement variability...")
    field_stats = analyzer.analyze_field_variability(field_data)
    field_stats.to_csv(output_dir / 'field_statistics.csv', index=False)
    print()

    # Compare stability
    print("Comparing anechoic vs field stability...")
    comparison = analyzer.compare_stability(anechoic_stats, field_stats)
    print()

    # Calculate fading statistics
    print("Calculating fading statistics...")
    fading_stats = analyzer.calculate_fading_statistics(field_data)
    fading_stats.to_csv(output_dir / 'fading_statistics.csv', index=False)
    print()

    # Generate visualizations
    print("Generating visualizations...")
    analyzer.plot_angular_pattern(anechoic_stats, output_dir)
    analyzer.plot_environment_comparison(anechoic_stats, field_stats, comparison, output_dir)
    analyzer.plot_fading_analysis(fading_stats, output_dir)
    print()

    # Generate report
    print("Generating analysis report...")
    analyzer.generate_report(anechoic_stats, field_stats, comparison, fading_stats, output_dir)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
