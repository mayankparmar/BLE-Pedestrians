"""
Body Shadowing Effects Analysis
================================
This script provides detailed quantification of body shadowing effects by comparing
LoS (Line-of-Sight) vs nLoS (Non-Line-of-Sight) RSSI measurements across different
distances and pathway positions.

Key Analyses:
1. Shadowing attenuation across distances (3m, 5m, 7m, 9m)
2. Position-dependent shadowing (start, mid_facing, center, mid_away, end)
3. Shadowing consistency and variability
4. Statistical significance testing
5. Predictive modeling for LoS/nLoS classification

Author: Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class BodyShadowingAnalyzer:
    """Analyzes body shadowing effects comparing LoS vs nLoS"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.distances = [3, 5, 7, 9]
        self.positions = ['start', 'mid_facing', 'center', 'mid_away', 'end']
        self.runs = [1, 2, 3]

        # Position mappings from A1
        self.position_map_los = {
            'start': 'start', 'mid_facing': 'mid_facing',
            'center': 'center', 'mid_away': 'mid_away', 'end': 'end'
        }
        self.position_map_nlos = {
            'start': 'start', 'mid_facing': 'mid facing',
            'center': 'centre', 'mid_away': 'mid away', 'end': 'end'
        }
        self.file_prefix_nlos = {
            'start': 'so', 'mid_facing': 'mfo',
            'center': 'co', 'mid_away': 'mao', 'end': 'eo'
        }

    def load_data(self, orientation):
        """Load deployment distance data"""
        data = {}
        position_map = self.position_map_los if orientation == 'los' else self.position_map_nlos

        for position in self.positions:
            data[position] = {}
            dir_name = position_map.get(position, position)
            position_path = self.dataset_path / orientation / dir_name

            if not position_path.exists():
                continue

            for distance in self.distances:
                data[position][distance] = {}

                for run in self.runs:
                    if orientation == 'los':
                        patterns = [
                            f"{position}_{distance}m_run{run}.csv",
                            f"run{run}/{position}_{distance}m_run{run}.csv"
                        ]
                    else:
                        prefix = self.file_prefix_nlos.get(position, position[:2])
                        patterns = [
                            f"{prefix}_{distance}m_run{run}.csv",
                            f"{prefix}_{distance}m_run{run}-1.csv"
                        ]

                    for pattern in patterns:
                        file_path = position_path / pattern
                        if file_path.exists():
                            try:
                                df = pd.read_csv(file_path)
                                data[position][distance][run] = df
                                break
                            except Exception as e:
                                print(f"Error loading {file_path}: {e}")

        return data

    def extract_rssi_statistics(self, data, orientation):
        """Extract RSSI statistics from loaded data"""
        results = []

        for position in data:
            for distance in data[position]:
                for run in data[position][distance]:
                    df = data[position][distance][run]

                    if 'rssi' in df.columns:
                        rssi_values = df['rssi'].dropna()
                    elif 'mean' in df.columns:
                        rssi_values = df['mean'].dropna()
                    else:
                        continue

                    if len(rssi_values) > 0:
                        results.append({
                            'orientation': orientation,
                            'distance': distance,
                            'position': position,
                            'run': run,
                            'mean_rssi': rssi_values.mean(),
                            'median_rssi': rssi_values.median(),
                            'std_rssi': rssi_values.std(),
                            'min_rssi': rssi_values.min(),
                            'max_rssi': rssi_values.max(),
                            'count': len(rssi_values)
                        })

        return pd.DataFrame(results)

    def calculate_shadowing_effect(self, los_stats, nlos_stats):
        """Calculate shadowing attenuation for each configuration"""
        results = []

        for distance in self.distances:
            for position in self.positions:
                for run in self.runs:
                    los_data = los_stats[
                        (los_stats['distance'] == distance) &
                        (los_stats['position'] == position) &
                        (los_stats['run'] == run)
                    ]

                    nlos_data = nlos_stats[
                        (nlos_stats['distance'] == distance) &
                        (nlos_stats['position'] == position) &
                        (nlos_stats['run'] == run)
                    ]

                    if not los_data.empty and not nlos_data.empty:
                        los_rssi = los_data['mean_rssi'].values[0]
                        nlos_rssi = nlos_data['mean_rssi'].values[0]
                        shadowing_db = los_rssi - nlos_rssi

                        results.append({
                            'distance': distance,
                            'position': position,
                            'run': run,
                            'los_rssi': los_rssi,
                            'nlos_rssi': nlos_rssi,
                            'shadowing_db': shadowing_db,
                            'los_std': los_data['std_rssi'].values[0],
                            'nlos_std': nlos_data['std_rssi'].values[0]
                        })

        return pd.DataFrame(results)

    def statistical_significance_test(self, los_stats, nlos_stats):
        """Perform statistical tests for LoS vs nLoS differences"""
        results = []

        for distance in self.distances:
            los_rssi = los_stats[los_stats['distance'] == distance]['mean_rssi'].values
            nlos_rssi = nlos_stats[nlos_stats['distance'] == distance]['mean_rssi'].values

            if len(los_rssi) > 1 and len(nlos_rssi) > 1:
                t_stat, p_value = ttest_ind(los_rssi, nlos_rssi)

                results.append({
                    'distance': distance,
                    'los_mean': np.mean(los_rssi),
                    'nlos_mean': np.mean(nlos_rssi),
                    'difference': np.mean(los_rssi) - np.mean(nlos_rssi),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

        return pd.DataFrame(results)

    def plot_shadowing_by_distance(self, shadowing_df, output_dir):
        """Visualize shadowing effect across distances"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Shadowing attenuation by distance
        ax1 = axes[0, 0]

        distances = []
        mean_shadowing = []
        std_shadowing = []

        for dist in self.distances:
            dist_data = shadowing_df[shadowing_df['distance'] == dist]
            if not dist_data.empty:
                distances.append(dist)
                mean_shadowing.append(dist_data['shadowing_db'].mean())
                std_shadowing.append(dist_data['shadowing_db'].std())

        ax1.errorbar(distances, mean_shadowing, yerr=std_shadowing,
                    fmt='o-', markersize=12, capsize=8, capthick=2,
                    linewidth=2.5, color='red', alpha=0.8)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Body Shadowing Attenuation (dB)', fontsize=12, fontweight='bold')
        ax1.set_title('Body Shadowing Effect vs Distance\n(Positive = LoS stronger than nLoS)',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(self.distances)

        # 2. LoS vs nLoS comparison
        ax2 = axes[0, 1]

        los_means = []
        nlos_means = []
        dist_labels = []

        for dist in self.distances:
            dist_data = shadowing_df[shadowing_df['distance'] == dist]
            if not dist_data.empty:
                los_means.append(dist_data['los_rssi'].mean())
                nlos_means.append(dist_data['nlos_rssi'].mean())
                dist_labels.append(dist)

        x = np.arange(len(dist_labels))
        width = 0.35

        bars1 = ax2.bar(x - width/2, los_means, width, label='LoS',
                       color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, nlos_means, width, label='nLoS',
                       color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)

        ax2.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean RSSI (dBm)', fontsize=12, fontweight='bold')
        ax2.set_title('LoS vs nLoS RSSI Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dist_labels)
        ax2.legend(fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)

        # 3. Position-dependent shadowing
        ax3 = axes[1, 0]

        position_labels = []
        position_shadowing = []

        for pos in self.positions:
            pos_data = shadowing_df[shadowing_df['position'] == pos]
            if not pos_data.empty:
                position_labels.append(pos.replace('_', '\n'))
                position_shadowing.append(pos_data['shadowing_db'].mean())

        colors = plt.cm.viridis(np.linspace(0, 1, len(position_labels)))
        bars = ax3.bar(range(len(position_labels)), position_shadowing,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_xlabel('Pathway Position', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Mean Shadowing Attenuation (dB)', fontsize=12, fontweight='bold')
        ax3.set_title('Body Shadowing by Pathway Position', fontsize=13, fontweight='bold')
        ax3.set_xticks(range(len(position_labels)))
        ax3.set_xticklabels(position_labels, fontsize=9)
        ax3.grid(True, axis='y', alpha=0.3)

        # 4. Shadowing variability
        ax4 = axes[1, 1]

        std_by_dist = []
        std_dist_labels = []
        for dist in self.distances:
            dist_data = shadowing_df[shadowing_df['distance'] == dist]
            if not dist_data.empty:
                std_by_dist.append(dist_data['shadowing_db'].std())
                std_dist_labels.append(dist)

        ax4.plot(std_dist_labels, std_by_dist, 'o-', markersize=12,
                linewidth=2.5, color='purple', alpha=0.8)
        ax4.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Shadowing Std Dev (dB)', fontsize=12, fontweight='bold')
        ax4.set_title('Shadowing Consistency vs Distance\n(Lower = More Consistent)',
                     fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(std_dist_labels)

        plt.tight_layout()
        plt.savefig(output_dir / 'body_shadowing_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'body_shadowing_analysis.pdf', bbox_inches='tight')
        print(f"Saved shadowing analysis plots to {output_dir}")
        plt.close()

    def plot_statistical_analysis(self, sig_test_df, output_dir):
        """Visualize statistical significance tests"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. P-value significance
        ax1 = axes[0]

        colors = ['green' if p < 0.05 else 'red' for p in sig_test_df['p_value']]
        bars = ax1.bar(range(len(sig_test_df)), sig_test_df['p_value'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax1.axhline(y=0.05, color='blue', linestyle='--', linewidth=2,
                   label='Significance Threshold (α=0.05)')
        ax1.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('P-value', fontsize=12, fontweight='bold')
        ax1.set_title('Statistical Significance of LoS vs nLoS Difference\n(Green = Significant, Red = Not Significant)',
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(sig_test_df)))
        ax1.set_xticklabels(sig_test_df['distance'].values)
        ax1.legend(fontsize=10)
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.set_yscale('log')

        # 2. Effect size (difference)
        ax2 = axes[1]

        ax2.bar(range(len(sig_test_df)), sig_test_df['difference'],
               color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1.5)
        ax2.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Difference (LoS - nLoS) [dB]', fontsize=12, fontweight='bold')
        ax2.set_title('Effect Size of Body Shadowing', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(sig_test_df)))
        ax2.set_xticklabels(sig_test_df['distance'].values)
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'statistical_significance.pdf', bbox_inches='tight')
        print(f"Saved statistical analysis plots to {output_dir}")
        plt.close()

    def generate_report(self, shadowing_df, sig_test_df, output_dir):
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("BODY SHADOWING EFFECTS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("OVERVIEW")
        report.append("-" * 80)
        report.append("This analysis quantifies the body shadowing effect by comparing RSSI")
        report.append("measurements in Line-of-Sight (LoS) vs Non-Line-of-Sight (nLoS) conditions.")
        report.append("Body shadowing occurs when the pedestrian's body blocks the direct signal")
        report.append("path between the broadcaster and observer.")
        report.append("")

        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        overall_mean = shadowing_df['shadowing_db'].mean()
        overall_std = shadowing_df['shadowing_db'].std()
        overall_min = shadowing_df['shadowing_db'].min()
        overall_max = shadowing_df['shadowing_db'].max()

        report.append(f"Overall Body Shadowing Attenuation:")
        report.append(f"  Mean: {overall_mean:.2f} ± {overall_std:.2f} dB")
        report.append(f"  Range: {overall_min:.2f} to {overall_max:.2f} dB")
        report.append(f"  Samples: {len(shadowing_df)}")
        report.append("")

        report.append("SHADOWING BY DISTANCE")
        report.append("-" * 80)
        for dist in self.distances:
            dist_data = shadowing_df[shadowing_df['distance'] == dist]
            if not dist_data.empty:
                mean_shad = dist_data['shadowing_db'].mean()
                std_shad = dist_data['shadowing_db'].std()
                los_mean = dist_data['los_rssi'].mean()
                nlos_mean = dist_data['nlos_rssi'].mean()

                report.append(f"\n{dist}m:")
                report.append(f"  LoS RSSI: {los_mean:.2f} dBm")
                report.append(f"  nLoS RSSI: {nlos_mean:.2f} dBm")
                report.append(f"  Shadowing: {mean_shad:.2f} ± {std_shad:.2f} dB")

        report.append("")
        report.append("SHADOWING BY POSITION")
        report.append("-" * 80)
        for pos in self.positions:
            pos_data = shadowing_df[shadowing_df['position'] == pos]
            if not pos_data.empty:
                mean_shad = pos_data['shadowing_db'].mean()
                std_shad = pos_data['shadowing_db'].std()

                report.append(f"\n{pos.replace('_', ' ').title()}:")
                report.append(f"  Shadowing: {mean_shad:.2f} ± {std_shad:.2f} dB")

        report.append("")
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-" * 80)
        report.append("Testing H0: LoS RSSI = nLoS RSSI (no shadowing effect)")
        report.append("Testing H1: LoS RSSI ≠ nLoS RSSI (shadowing effect exists)")
        report.append("")

        for _, row in sig_test_df.iterrows():
            report.append(f"{int(row['distance'])}m:")
            report.append(f"  t-statistic: {row['t_statistic']:.3f}")
            report.append(f"  p-value: {row['p_value']:.6f}")
            report.append(f"  Result: {'SIGNIFICANT' if row['significant'] else 'Not significant'} at α=0.05")
            report.append(f"  Effect size: {row['difference']:.2f} dB")
            report.append("")

        report.append("INTERPRETATION")
        report.append("-" * 80)

        if overall_mean > 0:
            report.append(f"Body shadowing causes an average attenuation of {overall_mean:.2f} dB,")
            report.append("meaning nLoS signals are weaker than LoS signals as expected.")
        else:
            report.append("WARNING: Negative shadowing detected - nLoS stronger than LoS.")
            report.append("This may indicate measurement issues or unexpected propagation effects.")

        report.append("")

        if overall_mean < 3:
            report.append("Shadowing effect is WEAK (< 3 dB) - may be difficult to detect reliably.")
        elif overall_mean < 6:
            report.append("Shadowing effect is MODERATE (3-6 dB) - detectable with averaging.")
        else:
            report.append("Shadowing effect is STRONG (> 6 dB) - easily detectable.")

        report.append("")

        if overall_std > 3:
            report.append(f"High variability (σ={overall_std:.2f} dB) suggests shadowing effect is")
            report.append("inconsistent across positions and distances. Classification may be unreliable.")
        else:
            report.append(f"Low variability (σ={overall_std:.2f} dB) suggests shadowing effect is")
            report.append("consistent. Good candidate for LoS/nLoS classification.")

        report.append("")
        report.append("=" * 80)

        # Save report
        report_text = "\n".join(report)
        with open(output_dir / 'body_shadowing_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {output_dir / 'body_shadowing_report.txt'}")


def main():
    """Main execution function"""

    base_path = Path("/home/user/BLE-Pedestrians")
    dataset_path = base_path / "dataset" / "Deployment Distance"
    output_dir = base_path / "analysis" / "A4_body_shadowing" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BODY SHADOWING EFFECTS ANALYSIS")
    print("=" * 80)
    print()

    analyzer = BodyShadowingAnalyzer(dataset_path)

    # Load data
    print("Loading LoS data...")
    los_data = analyzer.load_data('los')
    los_stats = analyzer.extract_rssi_statistics(los_data, 'los')
    print(f"  Loaded {len(los_stats)} LoS measurements")

    print("Loading nLoS data...")
    nlos_data = analyzer.load_data('nlos')
    nlos_stats = analyzer.extract_rssi_statistics(nlos_data, 'nlos')
    print(f"  Loaded {len(nlos_stats)} nLoS measurements")
    print()

    # Calculate shadowing effect
    print("Calculating body shadowing effects...")
    shadowing_df = analyzer.calculate_shadowing_effect(los_stats, nlos_stats)
    shadowing_df.to_csv(output_dir / 'shadowing_statistics.csv', index=False)
    print(f"  Calculated shadowing for {len(shadowing_df)} configurations")
    print()

    # Statistical significance testing
    print("Performing statistical significance tests...")
    sig_test_df = analyzer.statistical_significance_test(los_stats, nlos_stats)
    sig_test_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print()

    # Generate visualizations
    print("Generating visualizations...")
    analyzer.plot_shadowing_by_distance(shadowing_df, output_dir)
    analyzer.plot_statistical_analysis(sig_test_df, output_dir)
    print()

    # Generate report
    print("Generating analysis report...")
    analyzer.generate_report(shadowing_df, sig_test_df, output_dir)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
