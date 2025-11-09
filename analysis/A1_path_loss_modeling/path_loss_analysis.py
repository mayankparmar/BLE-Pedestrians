"""
Path Loss Modeling Analysis
===========================
This script analyzes BLE RSSI measurements at different deployment distances (3m, 5m, 7m, 9m)
to fit empirical path loss models and compare LoS vs nLoS propagation characteristics.

Path Loss Models:
1. Free-Space Path Loss (FSPL): PL(d) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
2. Log-Distance Path Loss: PL(d) = PL(d0) + 10*n*log10(d/d0) + X_σ
   where n is the path loss exponent and X_σ is shadowing term

Author: Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class PathLossAnalyzer:
    """Analyzes BLE path loss characteristics from deployment distance experiments"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.distances = [3, 5, 7, 9]  # meters
        self.positions = ['start', 'mid_facing', 'center', 'mid_away', 'end']
        # Position name mappings for different directory naming conventions
        self.position_map_los = {
            'start': 'start',
            'mid_facing': 'mid_facing',
            'center': 'center',
            'mid_away': 'mid_away',
            'end': 'end'
        }
        self.position_map_nlos = {
            'start': 'start',
            'mid_facing': 'mid facing',
            'center': 'centre',
            'mid_away': 'mid away',
            'end': 'end'
        }
        # File prefix mappings for nLoS data
        self.file_prefix_nlos = {
            'start': 'so',
            'mid_facing': 'mfo',
            'center': 'co',
            'mid_away': 'mao',
            'end': 'eo'
        }
        self.runs = [1, 2, 3]
        self.frequency = 2.4e9  # BLE operates at 2.4 GHz
        self.c = 3e8  # speed of light

    def load_deployment_data(self, orientation='los'):
        """
        Load all deployment distance data for a given orientation

        Args:
            orientation: 'los' or 'nlos'

        Returns:
            dict: Nested dictionary {position: {distance: {run: DataFrame}}}
        """
        data = {}

        # Select appropriate mapping
        if orientation == 'los':
            position_map = self.position_map_los
        else:
            position_map = self.position_map_nlos

        for position in self.positions:
            data[position] = {}
            # Use mapped directory name
            dir_name = position_map.get(position, position)
            position_path = self.dataset_path / orientation / dir_name

            if not position_path.exists():
                print(f"Warning: Path {position_path} does not exist")
                continue

            for distance in self.distances:
                data[position][distance] = {}

                for run in self.runs:
                    # Try multiple file patterns for LoS
                    if orientation == 'los':
                        patterns = [
                            f"{position}_{distance}m_run{run}.csv",
                            f"run{run}/{position}_{distance}m_run{run}.csv"
                        ]
                    else:
                        # nLoS uses different file prefixes
                        prefix = self.file_prefix_nlos.get(position, position[:2])
                        patterns = [
                            f"{prefix}_{distance}m_run{run}.csv",
                            f"{prefix}_{distance}m_run{run}-1.csv"  # Handle variants like "eo_9m_run3-1.csv"
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

    def extract_rssi_statistics(self, data):
        """
        Extract RSSI statistics for each distance

        Args:
            data: Nested dictionary from load_deployment_data

        Returns:
            DataFrame with columns: distance, position, run, mean_rssi, std_rssi, median_rssi, count
        """
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
                            'distance': distance,
                            'position': position,
                            'run': run,
                            'mean_rssi': rssi_values.mean(),
                            'median_rssi': rssi_values.median(),
                            'std_rssi': rssi_values.std(),
                            'min_rssi': rssi_values.min(),
                            'max_rssi': rssi_values.max(),
                            'count': len(rssi_values),
                            'q25': rssi_values.quantile(0.25),
                            'q75': rssi_values.quantile(0.75)
                        })

        return pd.DataFrame(results)

    def free_space_path_loss(self, distance):
        """
        Calculate theoretical Free-Space Path Loss (FSPL)

        FSPL(d) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)

        Args:
            distance: Distance in meters

        Returns:
            Path loss in dB
        """
        if distance <= 0:
            return np.inf

        # FSPL in dB
        fspl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) + \
               20 * np.log10(4 * np.pi / self.c)

        return fspl

    def log_distance_model(self, distance, pl_d0, n):
        """
        Log-Distance Path Loss Model

        PL(d) = PL(d0) + 10*n*log10(d/d0)

        Args:
            distance: Distance in meters
            pl_d0: Path loss at reference distance (d0 = 1m)
            n: Path loss exponent

        Returns:
            Path loss in dB
        """
        d0 = 1.0  # reference distance in meters
        return pl_d0 + 10 * n * np.log10(distance / d0)

    def fit_path_loss_model(self, stats_df):
        """
        Fit log-distance path loss model to measured data

        Args:
            stats_df: DataFrame with RSSI statistics

        Returns:
            dict with fitted parameters and goodness of fit metrics
        """
        # Check if stats_df is empty or has no distance column
        if stats_df is None or len(stats_df) == 0:
            print("Warning: Empty statistics DataFrame, cannot fit model")
            return None

        if 'distance' not in stats_df.columns:
            print("Warning: 'distance' column not found in DataFrame")
            return None

        # Aggregate data by distance (average across positions and runs)
        distance_stats = stats_df.groupby('distance').agg({
            'mean_rssi': ['mean', 'std', 'count'],
            'std_rssi': 'mean'
        }).reset_index()

        distance_stats.columns = ['distance', 'avg_rssi', 'rssi_std', 'n_samples', 'avg_std']

        # Convert RSSI to path loss (assuming Tx power, we'll estimate it)
        # PL = Tx_power - RSSI_measured
        # We'll fit to find Tx_power and path loss exponent

        distances = distance_stats['distance'].values
        measured_rssi = distance_stats['avg_rssi'].values

        # Fit log-distance model: RSSI(d) = RSSI(d0) - 10*n*log10(d/d0)
        # This is equivalent to: RSSI(d) = A - 10*n*log10(d)
        def rssi_model(d, A, n):
            return A - 10 * n * np.log10(d)

        try:
            # Initial guess: A (RSSI at 1m), n (path loss exponent, ~2 for free space)
            popt, pcov = curve_fit(rssi_model, distances, measured_rssi,
                                  p0=[-40, 2.0], maxfev=10000)

            A_fit, n_fit = popt
            perr = np.sqrt(np.diag(pcov))

            # Calculate R-squared
            predicted_rssi = rssi_model(distances, A_fit, n_fit)
            ss_res = np.sum((measured_rssi - predicted_rssi) ** 2)
            ss_tot = np.sum((measured_rssi - np.mean(measured_rssi)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate RMSE
            rmse = np.sqrt(np.mean((measured_rssi - predicted_rssi) ** 2))

            return {
                'rssi_at_1m': A_fit,
                'rssi_at_1m_std': perr[0],
                'path_loss_exponent': n_fit,
                'path_loss_exponent_std': perr[1],
                'r_squared': r_squared,
                'rmse': rmse,
                'distances': distances,
                'measured_rssi': measured_rssi,
                'predicted_rssi': predicted_rssi,
                'distance_stats': distance_stats
            }
        except Exception as e:
            print(f"Error fitting model: {e}")
            return None

    def compare_positions(self, stats_df):
        """
        Compare RSSI measurements across different positions along the pathway

        Args:
            stats_df: DataFrame with RSSI statistics

        Returns:
            DataFrame with position-wise statistics
        """
        position_comparison = stats_df.groupby(['position', 'distance']).agg({
            'mean_rssi': ['mean', 'std'],
            'std_rssi': 'mean',
            'count': 'sum'
        }).reset_index()

        return position_comparison

    def plot_path_loss(self, los_fit, nlos_fit, output_dir):
        """
        Create comprehensive path loss visualization

        Args:
            los_fit: Fitted model results for LoS
            nlos_fit: Fitted model results for nLoS
            output_dir: Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Path Loss vs Distance (LoS)
        ax1 = axes[0, 0]
        if los_fit:
            distances_fine = np.linspace(2.5, 10, 100)

            # Measured data
            ax1.errorbar(los_fit['distances'], los_fit['measured_rssi'],
                        yerr=los_fit['distance_stats']['rssi_std'],
                        fmt='o', markersize=8, capsize=5, capthick=2,
                        label='Measured (LoS)', color='blue', alpha=0.7)

            # Fitted model
            fitted_rssi = los_fit['rssi_at_1m'] - 10 * los_fit['path_loss_exponent'] * np.log10(distances_fine)
            ax1.plot(distances_fine, fitted_rssi, '--', linewidth=2,
                    label=f'Fitted: n={los_fit["path_loss_exponent"]:.2f}', color='blue')

            # Free-space model (n=2)
            free_space_rssi = los_fit['rssi_at_1m'] - 10 * 2.0 * np.log10(distances_fine)
            ax1.plot(distances_fine, free_space_rssi, ':', linewidth=2,
                    label='Free-space (n=2)', color='green')

            ax1.set_xlabel('Distance (m)', fontsize=12)
            ax1.set_ylabel('RSSI (dBm)', fontsize=12)
            ax1.set_title(f'LoS Path Loss Model\n(R²={los_fit["r_squared"]:.3f}, RMSE={los_fit["rmse"]:.2f} dB)',
                         fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

        # 2. Path Loss vs Distance (nLoS)
        ax2 = axes[0, 1]
        if nlos_fit:
            # Measured data
            ax2.errorbar(nlos_fit['distances'], nlos_fit['measured_rssi'],
                        yerr=nlos_fit['distance_stats']['rssi_std'],
                        fmt='s', markersize=8, capsize=5, capthick=2,
                        label='Measured (nLoS)', color='red', alpha=0.7)

            # Fitted model
            fitted_rssi = nlos_fit['rssi_at_1m'] - 10 * nlos_fit['path_loss_exponent'] * np.log10(distances_fine)
            ax2.plot(distances_fine, fitted_rssi, '--', linewidth=2,
                    label=f'Fitted: n={nlos_fit["path_loss_exponent"]:.2f}', color='red')

            # Free-space model (n=2)
            free_space_rssi = nlos_fit['rssi_at_1m'] - 10 * 2.0 * np.log10(distances_fine)
            ax2.plot(distances_fine, free_space_rssi, ':', linewidth=2,
                    label='Free-space (n=2)', color='green')

            ax2.set_xlabel('Distance (m)', fontsize=12)
            ax2.set_ylabel('RSSI (dBm)', fontsize=12)
            ax2.set_title(f'nLoS Path Loss Model\n(R²={nlos_fit["r_squared"]:.3f}, RMSE={nlos_fit["rmse"]:.2f} dB)',
                         fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # 3. LoS vs nLoS Comparison
        ax3 = axes[1, 0]
        if los_fit and nlos_fit:
            ax3.plot(los_fit['distances'], los_fit['measured_rssi'], 'o-',
                    markersize=8, linewidth=2, label='LoS', color='blue', alpha=0.7)
            ax3.plot(nlos_fit['distances'], nlos_fit['measured_rssi'], 's-',
                    markersize=8, linewidth=2, label='nLoS', color='red', alpha=0.7)

            # Add error bars
            ax3.fill_between(los_fit['distances'],
                            los_fit['measured_rssi'] - los_fit['distance_stats']['rssi_std'],
                            los_fit['measured_rssi'] + los_fit['distance_stats']['rssi_std'],
                            alpha=0.2, color='blue')
            ax3.fill_between(nlos_fit['distances'],
                            nlos_fit['measured_rssi'] - nlos_fit['distance_stats']['rssi_std'],
                            nlos_fit['measured_rssi'] + nlos_fit['distance_stats']['rssi_std'],
                            alpha=0.2, color='red')

            ax3.set_xlabel('Distance (m)', fontsize=12)
            ax3.set_ylabel('RSSI (dBm)', fontsize=12)
            ax3.set_title('LoS vs nLoS Comparison', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

        # 4. Path Loss Exponent Comparison
        ax4 = axes[1, 1]
        if los_fit and nlos_fit:
            exponents = [los_fit['path_loss_exponent'], nlos_fit['path_loss_exponent'], 2.0]
            errors = [los_fit['path_loss_exponent_std'], nlos_fit['path_loss_exponent_std'], 0]
            labels = ['LoS\n(Measured)', 'nLoS\n(Measured)', 'Free-Space\n(Theoretical)']
            colors = ['blue', 'red', 'green']

            bars = ax4.bar(labels, exponents, yerr=errors, capsize=10,
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            ax4.axhline(y=2.0, color='gray', linestyle='--', linewidth=2,
                       label='Free-space reference (n=2)')
            ax4.set_ylabel('Path Loss Exponent (n)', fontsize=12)
            ax4.set_title('Path Loss Exponent Comparison', fontsize=13, fontweight='bold')
            ax4.grid(True, axis='y', alpha=0.3)
            ax4.legend(fontsize=9)

            # Add value labels on bars
            for i, (bar, val, err) in enumerate(zip(bars, exponents, errors)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + err + 0.1,
                        f'{val:.2f}±{err:.2f}' if err > 0 else f'{val:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'path_loss_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'path_loss_analysis.pdf', bbox_inches='tight')
        print(f"Saved path loss plots to {output_dir}")
        plt.close()

    def plot_position_analysis(self, los_stats, nlos_stats, output_dir):
        """
        Analyze and plot RSSI variation across different positions

        Args:
            los_stats: LoS statistics DataFrame
            nlos_stats: nLoS statistics DataFrame
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Position order for plotting
        position_order = ['start', 'mid_facing', 'center', 'mid_away', 'end']

        for idx, distance in enumerate([3, 5, 7, 9]):
            ax = axes[idx // 2, idx % 2]

            # Filter data for this distance
            los_dist = los_stats[los_stats['distance'] == distance]
            nlos_dist = nlos_stats[nlos_stats['distance'] == distance]

            # Prepare data for plotting
            los_means = []
            nlos_means = []
            los_stds = []
            nlos_stds = []

            for pos in position_order:
                los_pos = los_dist[los_dist['position'] == pos]['mean_rssi']
                nlos_pos = nlos_dist[nlos_dist['position'] == pos]['mean_rssi']

                los_means.append(los_pos.mean() if len(los_pos) > 0 else np.nan)
                nlos_means.append(nlos_pos.mean() if len(nlos_pos) > 0 else np.nan)
                los_stds.append(los_pos.std() if len(los_pos) > 0 else 0)
                nlos_stds.append(nlos_pos.std() if len(nlos_pos) > 0 else 0)

            x = np.arange(len(position_order))
            width = 0.35

            ax.bar(x - width/2, los_means, width, yerr=los_stds, label='LoS',
                  color='blue', alpha=0.7, capsize=5)
            ax.bar(x + width/2, nlos_means, width, yerr=nlos_stds, label='nLoS',
                  color='red', alpha=0.7, capsize=5)

            ax.set_xlabel('Position', fontsize=11)
            ax.set_ylabel('Mean RSSI (dBm)', fontsize=11)
            ax.set_title(f'RSSI by Position (Distance = {distance}m)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([p.replace('_', '\n') for p in position_order], fontsize=9)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'position_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved position analysis plot to {output_dir}")
        plt.close()

    def generate_report(self, los_fit, nlos_fit, output_dir):
        """
        Generate a comprehensive text report of the analysis

        Args:
            los_fit: LoS fitting results
            nlos_fit: nLoS fitting results
            output_dir: Output directory
        """
        report = []
        report.append("=" * 80)
        report.append("BLE PATH LOSS MODELING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("EXPERIMENTAL SETUP")
        report.append("-" * 80)
        report.append(f"Deployment Distances: {self.distances} meters")
        report.append(f"Positions: {', '.join(self.positions)}")
        report.append(f"Runs per configuration: {len(self.runs)}")
        report.append(f"BLE Frequency: {self.frequency/1e9:.1f} GHz")
        report.append("")

        if los_fit:
            report.append("LINE-OF-SIGHT (LoS) RESULTS")
            report.append("-" * 80)
            report.append(f"RSSI at 1m (reference): {los_fit['rssi_at_1m']:.2f} ± {los_fit['rssi_at_1m_std']:.2f} dBm")
            report.append(f"Path Loss Exponent (n): {los_fit['path_loss_exponent']:.3f} ± {los_fit['path_loss_exponent_std']:.3f}")
            report.append(f"R-squared: {los_fit['r_squared']:.4f}")
            report.append(f"RMSE: {los_fit['rmse']:.3f} dB")
            report.append("")
            report.append("Measured vs Predicted RSSI:")
            for i, d in enumerate(los_fit['distances']):
                measured = los_fit['measured_rssi'][i]
                predicted = los_fit['predicted_rssi'][i]
                error = measured - predicted
                report.append(f"  {d}m: Measured = {measured:.2f} dBm, Predicted = {predicted:.2f} dBm, Error = {error:.2f} dB")
            report.append("")

        if nlos_fit:
            report.append("NON-LINE-OF-SIGHT (nLoS) RESULTS")
            report.append("-" * 80)
            report.append(f"RSSI at 1m (reference): {nlos_fit['rssi_at_1m']:.2f} ± {nlos_fit['rssi_at_1m_std']:.2f} dBm")
            report.append(f"Path Loss Exponent (n): {nlos_fit['path_loss_exponent']:.3f} ± {nlos_fit['path_loss_exponent_std']:.3f}")
            report.append(f"R-squared: {nlos_fit['r_squared']:.4f}")
            report.append(f"RMSE: {nlos_fit['rmse']:.3f} dB")
            report.append("")
            report.append("Measured vs Predicted RSSI:")
            for i, d in enumerate(nlos_fit['distances']):
                measured = nlos_fit['measured_rssi'][i]
                predicted = nlos_fit['predicted_rssi'][i]
                error = measured - predicted
                report.append(f"  {d}m: Measured = {measured:.2f} dBm, Predicted = {predicted:.2f} dBm, Error = {error:.2f} dB")
            report.append("")

        if los_fit and nlos_fit:
            report.append("COMPARATIVE ANALYSIS")
            report.append("-" * 80)
            n_diff = nlos_fit['path_loss_exponent'] - los_fit['path_loss_exponent']
            report.append(f"Path Loss Exponent Difference (nLoS - LoS): {n_diff:.3f}")
            report.append(f"  - LoS exponent: {los_fit['path_loss_exponent']:.3f}")
            report.append(f"  - nLoS exponent: {nlos_fit['path_loss_exponent']:.3f}")
            report.append(f"  - Free-space reference: 2.000")
            report.append("")

            report.append("Body Shadowing Effect (RSSI difference at each distance):")
            # Find common distances
            los_distances = list(los_fit['distances'])
            nlos_distances = list(nlos_fit['distances'])
            common_distances = sorted(set(los_distances).intersection(set(nlos_distances)))

            shadowing_diffs = []
            for d in common_distances:
                los_idx = los_distances.index(d)
                nlos_idx = nlos_distances.index(d)
                los_rssi = los_fit['measured_rssi'][los_idx]
                nlos_rssi = nlos_fit['measured_rssi'][nlos_idx]
                diff = los_rssi - nlos_rssi
                shadowing_diffs.append(diff)
                report.append(f"  {d}m: LoS = {los_rssi:.2f} dBm, nLoS = {nlos_rssi:.2f} dBm, Δ = {diff:.2f} dB")
            report.append("")

            avg_shadowing = np.mean(shadowing_diffs) if shadowing_diffs else 0
            report.append(f"Average Body Shadowing Effect: {avg_shadowing:.2f} dB")
            report.append("")

        report.append("INTERPRETATION")
        report.append("-" * 80)
        if los_fit:
            if los_fit['path_loss_exponent'] < 2.0:
                report.append("LoS: Path loss exponent < 2 suggests waveguide or ground reflection effects")
            elif los_fit['path_loss_exponent'] > 2.5:
                report.append("LoS: Path loss exponent > 2.5 suggests additional attenuation factors")
            else:
                report.append("LoS: Path loss exponent close to free-space value (2.0)")

        if nlos_fit:
            if nlos_fit['path_loss_exponent'] > 3.0:
                report.append("nLoS: High path loss exponent indicates significant body shadowing")
            elif nlos_fit['path_loss_exponent'] > 2.5:
                report.append("nLoS: Moderate body shadowing effect observed")
            else:
                report.append("nLoS: Limited body shadowing effect")

        report.append("")
        report.append("=" * 80)

        # Save report
        report_text = "\n".join(report)
        with open(output_dir / 'path_loss_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {output_dir / 'path_loss_report.txt'}")

        return report_text


def main():
    """Main execution function"""

    # Setup paths
    base_path = Path("/home/user/BLE-Pedestrians")
    dataset_path = base_path / "dataset" / "Deployment Distance"
    output_dir = base_path / "analysis" / "A1_path_loss_modeling" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BLE PATH LOSS MODELING ANALYSIS")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = PathLossAnalyzer(dataset_path)

    # Load LoS data
    print("Loading LoS data...")
    los_data = analyzer.load_deployment_data('los')
    los_stats = analyzer.extract_rssi_statistics(los_data)
    print(f"  Loaded {len(los_stats)} LoS measurements")
    los_stats.to_csv(output_dir / 'los_statistics.csv', index=False)

    # Load nLoS data
    print("Loading nLoS data...")
    nlos_data = analyzer.load_deployment_data('nlos')
    nlos_stats = analyzer.extract_rssi_statistics(nlos_data)
    print(f"  Loaded {len(nlos_stats)} nLoS measurements")
    nlos_stats.to_csv(output_dir / 'nlos_statistics.csv', index=False)
    print()

    # Fit path loss models
    print("Fitting path loss models...")
    los_fit = analyzer.fit_path_loss_model(los_stats)
    nlos_fit = analyzer.fit_path_loss_model(nlos_stats)
    print()

    # Generate visualizations
    print("Generating visualizations...")
    analyzer.plot_path_loss(los_fit, nlos_fit, output_dir)
    analyzer.plot_position_analysis(los_stats, nlos_stats, output_dir)
    print()

    # Generate report
    print("Generating analysis report...")
    analyzer.generate_report(los_fit, nlos_fit, output_dir)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
