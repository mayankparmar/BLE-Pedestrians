"""
Advertisement Interval Impact Analysis
======================================
This script analyzes the impact of BLE advertisement intervals (100ms, 500ms, 1000ms)
on packet reception, detection latency, signal quality, and reliability.

Key Metrics:
1. Packet Reception Rate (PRR): Percentage of expected packets received
2. Detection Latency: Time to first detection
3. Signal Quality: RSSI mean and standard deviation
4. Detection Reliability: Consistency of detection across runs

Author: Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class AdvertisementIntervalAnalyzer:
    """Analyzes the impact of different BLE advertisement intervals"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.intervals = [100, 500, 1000]  # milliseconds
        self.orientations = ['los', 'nlos']
        self.direction_codes = ['los_es', 'los_se', 'nlos_es', 'nlos_se']

    def load_interval_data(self, interval, orientation_code):
        """
        Load data for a specific interval and orientation

        Args:
            interval: Advertisement interval in ms (100, 500, 1000)
            orientation_code: Direction code (los_es, los_se, nlos_es, nlos_se)

        Returns:
            dict with 'observer', 'broadcaster', 'gps' DataFrames
        """
        base_path = self.dataset_path / "processed data" / "3m" / f"{interval}ms" / orientation_code

        data = {}

        # Load observer data (RSSI measurements)
        observer_file = base_path / "observer.csv"
        if observer_file.exists():
            data['observer'] = pd.read_csv(observer_file)
            # Convert time to datetime if it's not already
            if 'time' in data['observer'].columns:
                data['observer']['time'] = pd.to_datetime(data['observer']['time'])
        else:
            print(f"Warning: Observer file not found: {observer_file}")
            data['observer'] = pd.DataFrame()

        # Load broadcaster data (transmission records)
        broadcaster_file = base_path / "broadcaster.csv"
        if broadcaster_file.exists():
            data['broadcaster'] = pd.read_csv(broadcaster_file)
            if 'time' in data['broadcaster'].columns:
                data['broadcaster']['time'] = pd.to_datetime(data['broadcaster']['time'])
        else:
            print(f"Warning: Broadcaster file not found: {broadcaster_file}")
            data['broadcaster'] = pd.DataFrame()

        # Load GPS data
        gps_file = base_path / "gps.csv"
        if gps_file.exists():
            data['gps'] = pd.read_csv(gps_file)
            if 'time_' in data['gps'].columns:
                data['gps']['time'] = pd.to_datetime(data['gps']['time_'])
            elif 'time' in data['gps'].columns:
                data['gps']['time'] = pd.to_datetime(data['gps']['time'])
        else:
            print(f"Warning: GPS file not found: {gps_file}")
            data['gps'] = pd.DataFrame()

        return data

    def calculate_packet_reception_rate(self, observer_df, broadcaster_df):
        """
        Calculate Packet Reception Rate (PRR)

        PRR = (Number of received packets / Number of transmitted packets) * 100

        Args:
            observer_df: Observer (receiver) DataFrame
            broadcaster_df: Broadcaster (transmitter) DataFrame

        Returns:
            dict with PRR metrics
        """
        if broadcaster_df.empty or observer_df.empty:
            return None

        # Count transmitted packets
        n_transmitted = len(broadcaster_df)

        # Count received packets (unique sequence numbers in observer data)
        # The observer data shows cumulative statistics, so we need actual packet count
        n_received = len(observer_df)

        prr = (n_received / n_transmitted * 100) if n_transmitted > 0 else 0

        return {
            'transmitted': n_transmitted,
            'received': n_received,
            'prr_percent': prr,
            'packet_loss_percent': 100 - prr
        }

    def calculate_detection_latency(self, observer_df, broadcaster_df):
        """
        Calculate detection latency - time from first transmission to first reception

        Args:
            observer_df: Observer DataFrame
            broadcaster_df: Broadcaster DataFrame

        Returns:
            dict with latency metrics
        """
        if observer_df.empty or broadcaster_df.empty:
            return None

        if 'time' not in observer_df.columns or 'time' not in broadcaster_df.columns:
            return None

        first_tx = broadcaster_df['time'].min()
        first_rx = observer_df['time'].min()

        latency = (first_rx - first_tx).total_seconds()

        return {
            'first_transmission': first_tx,
            'first_reception': first_rx,
            'latency_seconds': latency,
            'latency_ms': latency * 1000
        }

    def calculate_signal_quality(self, observer_df):
        """
        Calculate signal quality metrics from RSSI measurements

        Args:
            observer_df: Observer DataFrame with RSSI data

        Returns:
            dict with signal quality metrics
        """
        if observer_df.empty:
            return None

        # Use the 'rssi' column for individual measurements
        # and 'mean', 'sd' for running statistics
        metrics = {}

        if 'rssi' in observer_df.columns:
            rssi_values = observer_df['rssi'].dropna()
            metrics['rssi_mean'] = rssi_values.mean()
            metrics['rssi_std'] = rssi_values.std()
            metrics['rssi_min'] = rssi_values.min()
            metrics['rssi_max'] = rssi_values.max()
            metrics['rssi_median'] = rssi_values.median()
            metrics['rssi_range'] = rssi_values.max() - rssi_values.min()

        if 'mean' in observer_df.columns:
            metrics['running_mean_avg'] = observer_df['mean'].mean()

        if 'sd' in observer_df.columns:
            metrics['running_std_avg'] = observer_df['sd'].mean()
            metrics['stability_score'] = 1 / (1 + observer_df['sd'].mean())  # Higher is better

        return metrics

    def calculate_detection_gaps(self, observer_df, expected_interval_ms):
        """
        Calculate gaps in detection (missed packets)

        Args:
            observer_df: Observer DataFrame
            expected_interval_ms: Expected interval in milliseconds

        Returns:
            dict with gap statistics
        """
        if observer_df.empty or 'time' not in observer_df.columns:
            return None

        # Calculate time differences between consecutive receptions
        observer_df = observer_df.sort_values('time')
        time_diffs = observer_df['time'].diff().dt.total_seconds() * 1000  # Convert to ms

        # Remove NaN (first value)
        time_diffs = time_diffs.dropna()

        if len(time_diffs) == 0:
            return None

        # A gap is when time difference > expected_interval * threshold
        threshold = 1.5
        gaps = time_diffs[time_diffs > expected_interval_ms * threshold]

        return {
            'n_gaps': len(gaps),
            'gap_rate': len(gaps) / len(time_diffs) * 100 if len(time_diffs) > 0 else 0,
            'avg_gap_duration_ms': gaps.mean() if len(gaps) > 0 else 0,
            'max_gap_duration_ms': gaps.max() if len(gaps) > 0 else 0,
            'avg_inter_packet_interval_ms': time_diffs.mean(),
            'std_inter_packet_interval_ms': time_diffs.std()
        }

    def analyze_all_intervals(self):
        """
        Comprehensive analysis across all intervals and orientations

        Returns:
            DataFrame with all metrics
        """
        results = []

        for interval in self.intervals:
            for direction_code in self.direction_codes:
                print(f"Analyzing {interval}ms - {direction_code}...")

                data = self.load_interval_data(interval, direction_code)

                if data['observer'].empty:
                    continue

                # Extract orientation (los/nlos) and direction (es/se)
                orientation = 'los' if 'los' in direction_code else 'nlos'
                direction = direction_code.split('_')[1]

                # Calculate metrics
                prr = self.calculate_packet_reception_rate(data['observer'], data['broadcaster'])
                latency = self.calculate_detection_latency(data['observer'], data['broadcaster'])
                quality = self.calculate_signal_quality(data['observer'])
                gaps = self.calculate_detection_gaps(data['observer'], interval)

                # Compile results
                result = {
                    'interval_ms': interval,
                    'orientation': orientation,
                    'direction': direction,
                    'direction_code': direction_code
                }

                if prr:
                    result.update({f'prr_{k}': v for k, v in prr.items()})
                if latency:
                    result.update({f'latency_{k}': v for k, v in latency.items()})
                if quality:
                    result.update({f'quality_{k}': v for k, v in quality.items()})
                if gaps:
                    result.update({f'gap_{k}': v for k, v in gaps.items()})

                results.append(result)

        return pd.DataFrame(results)

    def plot_packet_reception_analysis(self, results_df, output_dir):
        """
        Visualize packet reception rate analysis

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. PRR by Interval
        ax1 = axes[0, 0]
        for orientation in ['los', 'nlos']:
            data = results_df[results_df['orientation'] == orientation]
            if not data.empty:
                prr_means = data.groupby('interval_ms')['prr_prr_percent'].mean()
                prr_stds = data.groupby('interval_ms')['prr_prr_percent'].std()

                ax1.errorbar(prr_means.index, prr_means.values, yerr=prr_stds.values,
                           marker='o', markersize=10, capsize=5, capthick=2,
                           label=orientation.upper(), linewidth=2, alpha=0.8)

        ax1.set_xlabel('Advertisement Interval (ms)', fontsize=12)
        ax1.set_ylabel('Packet Reception Rate (%)', fontsize=12)
        ax1.set_title('Packet Reception Rate vs Advertisement Interval', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])

        # 2. Packet Loss by Interval
        ax2 = axes[0, 1]
        for orientation in ['los', 'nlos']:
            data = results_df[results_df['orientation'] == orientation]
            if not data.empty:
                loss_means = data.groupby('interval_ms')['prr_packet_loss_percent'].mean()
                loss_stds = data.groupby('interval_ms')['prr_packet_loss_percent'].std()

                ax2.errorbar(loss_means.index, loss_means.values, yerr=loss_stds.values,
                           marker='s', markersize=10, capsize=5, capthick=2,
                           label=orientation.upper(), linewidth=2, alpha=0.8)

        ax2.set_xlabel('Advertisement Interval (ms)', fontsize=12)
        ax2.set_ylabel('Packet Loss Rate (%)', fontsize=12)
        ax2.set_title('Packet Loss Rate vs Advertisement Interval', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 3. Detection Latency by Interval
        ax3 = axes[1, 0]
        if 'latency_latency_ms' in results_df.columns:
            for orientation in ['los', 'nlos']:
                data = results_df[results_df['orientation'] == orientation]
                if not data.empty:
                    latency_means = data.groupby('interval_ms')['latency_latency_ms'].mean()
                    latency_stds = data.groupby('interval_ms')['latency_latency_ms'].std()

                    ax3.errorbar(latency_means.index, latency_means.values, yerr=latency_stds.values,
                               marker='^', markersize=10, capsize=5, capthick=2,
                               label=orientation.upper(), linewidth=2, alpha=0.8)

            ax3.set_xlabel('Advertisement Interval (ms)', fontsize=12)
            ax3.set_ylabel('Detection Latency (ms)', fontsize=12)
            ax3.set_title('Detection Latency vs Advertisement Interval', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)

        # 4. Gap Rate by Interval
        ax4 = axes[1, 1]
        if 'gap_gap_rate' in results_df.columns:
            for orientation in ['los', 'nlos']:
                data = results_df[results_df['orientation'] == orientation]
                if not data.empty:
                    gap_means = data.groupby('interval_ms')['gap_gap_rate'].mean()
                    gap_stds = data.groupby('interval_ms')['gap_gap_rate'].std()

                    ax4.errorbar(gap_means.index, gap_means.values, yerr=gap_stds.values,
                               marker='d', markersize=10, capsize=5, capthick=2,
                               label=orientation.upper(), linewidth=2, alpha=0.8)

            ax4.set_xlabel('Advertisement Interval (ms)', fontsize=12)
            ax4.set_ylabel('Detection Gap Rate (%)', fontsize=12)
            ax4.set_title('Detection Gap Rate vs Advertisement Interval', fontsize=13, fontweight='bold')
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'packet_reception_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'packet_reception_analysis.pdf', bbox_inches='tight')
        print(f"Saved packet reception plots to {output_dir}")
        plt.close()

    def plot_signal_quality_analysis(self, results_df, output_dir):
        """
        Visualize signal quality metrics

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. RSSI Mean by Interval
        ax1 = axes[0, 0]
        if 'quality_rssi_mean' in results_df.columns:
            for orientation in ['los', 'nlos']:
                data = results_df[results_df['orientation'] == orientation]
                if not data.empty:
                    rssi_means = data.groupby('interval_ms')['quality_rssi_mean'].mean()
                    rssi_stds = data.groupby('interval_ms')['quality_rssi_mean'].std()

                    ax1.errorbar(rssi_means.index, rssi_means.values, yerr=rssi_stds.values,
                               marker='o', markersize=10, capsize=5, capthick=2,
                               label=orientation.upper(), linewidth=2, alpha=0.8)

            ax1.set_xlabel('Advertisement Interval (ms)', fontsize=12)
            ax1.set_ylabel('Mean RSSI (dBm)', fontsize=12)
            ax1.set_title('Signal Strength vs Advertisement Interval', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

        # 2. RSSI Stability (Std Dev) by Interval
        ax2 = axes[0, 1]
        if 'quality_rssi_std' in results_df.columns:
            for orientation in ['los', 'nlos']:
                data = results_df[results_df['orientation'] == orientation]
                if not data.empty:
                    std_means = data.groupby('interval_ms')['quality_rssi_std'].mean()
                    std_stds = data.groupby('interval_ms')['quality_rssi_std'].std()

                    ax2.errorbar(std_means.index, std_means.values, yerr=std_stds.values,
                               marker='s', markersize=10, capsize=5, capthick=2,
                               label=orientation.upper(), linewidth=2, alpha=0.8)

            ax2.set_xlabel('Advertisement Interval (ms)', fontsize=12)
            ax2.set_ylabel('RSSI Standard Deviation (dB)', fontsize=12)
            ax2.set_title('Signal Stability vs Advertisement Interval\n(Lower is More Stable)',
                         fontsize=13, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

        # 3. RSSI Range by Interval
        ax3 = axes[1, 0]
        if 'quality_rssi_range' in results_df.columns:
            for orientation in ['los', 'nlos']:
                data = results_df[results_df['orientation'] == orientation]
                if not data.empty:
                    range_means = data.groupby('interval_ms')['quality_rssi_range'].mean()
                    range_stds = data.groupby('interval_ms')['quality_rssi_range'].std()

                    ax3.errorbar(range_means.index, range_means.values, yerr=range_stds.values,
                               marker='^', markersize=10, capsize=5, capthick=2,
                               label=orientation.upper(), linewidth=2, alpha=0.8)

            ax3.set_xlabel('Advertisement Interval (ms)', fontsize=12)
            ax3.set_ylabel('RSSI Range (dB)', fontsize=12)
            ax3.set_title('Signal Variability Range vs Advertisement Interval', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)

        # 4. Inter-Packet Interval Consistency
        ax4 = axes[1, 1]
        if 'gap_std_inter_packet_interval_ms' in results_df.columns:
            for orientation in ['los', 'nlos']:
                data = results_df[results_df['orientation'] == orientation]
                if not data.empty:
                    ipi_means = data.groupby('interval_ms')['gap_std_inter_packet_interval_ms'].mean()
                    ipi_stds = data.groupby('interval_ms')['gap_std_inter_packet_interval_ms'].std()

                    ax4.errorbar(ipi_means.index, ipi_means.values, yerr=ipi_stds.values,
                               marker='d', markersize=10, capsize=5, capthick=2,
                               label=orientation.upper(), linewidth=2, alpha=0.8)

            ax4.set_xlabel('Advertisement Interval (ms)', fontsize=12)
            ax4.set_ylabel('Inter-Packet Interval Std Dev (ms)', fontsize=12)
            ax4.set_title('Timing Consistency vs Advertisement Interval\n(Lower is More Consistent)',
                         fontsize=13, fontweight='bold')
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'signal_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'signal_quality_analysis.pdf', bbox_inches='tight')
        print(f"Saved signal quality plots to {output_dir}")
        plt.close()

    def generate_report(self, results_df, output_dir):
        """
        Generate comprehensive analysis report

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
        """
        report = []
        report.append("=" * 80)
        report.append("BLE ADVERTISEMENT INTERVAL IMPACT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("EXPERIMENTAL SETUP")
        report.append("-" * 80)
        report.append(f"Advertisement Intervals Tested: {self.intervals} ms")
        report.append(f"Orientations: LoS and nLoS")
        report.append(f"Deployment Distance: 3 meters")
        report.append(f"Walking Directions: Both (es and se)")
        report.append("")

        # Aggregate by interval
        for interval in self.intervals:
            report.append(f"\n{'=' * 80}")
            report.append(f"INTERVAL: {interval} ms")
            report.append("=" * 80)

            interval_data = results_df[results_df['interval_ms'] == interval]

            for orientation in ['los', 'nlos']:
                orient_data = interval_data[interval_data['orientation'] == orientation]

                if orient_data.empty:
                    continue

                report.append(f"\n{orientation.upper()} Configuration:")
                report.append("-" * 40)

                # Packet Reception
                if 'prr_prr_percent' in orient_data.columns:
                    prr_mean = orient_data['prr_prr_percent'].mean()
                    prr_std = orient_data['prr_prr_percent'].std()
                    report.append(f"  Packet Reception Rate: {prr_mean:.2f}% ± {prr_std:.2f}%")

                    loss_mean = orient_data['prr_packet_loss_percent'].mean()
                    report.append(f"  Packet Loss Rate: {loss_mean:.2f}%")

                # Detection Latency
                if 'latency_latency_ms' in orient_data.columns:
                    latency_mean = orient_data['latency_latency_ms'].mean()
                    latency_std = orient_data['latency_latency_ms'].std()
                    report.append(f"  Detection Latency: {latency_mean:.2f} ± {latency_std:.2f} ms")

                # Signal Quality
                if 'quality_rssi_mean' in orient_data.columns:
                    rssi_mean = orient_data['quality_rssi_mean'].mean()
                    rssi_std_mean = orient_data['quality_rssi_std'].mean()
                    report.append(f"  Mean RSSI: {rssi_mean:.2f} dBm")
                    report.append(f"  RSSI Stability (Std Dev): {rssi_std_mean:.2f} dB")

                # Gaps
                if 'gap_gap_rate' in orient_data.columns:
                    gap_rate = orient_data['gap_gap_rate'].mean()
                    report.append(f"  Detection Gap Rate: {gap_rate:.2f}%")

                report.append("")

        # Comparative Summary
        report.append("\n" + "=" * 80)
        report.append("COMPARATIVE SUMMARY")
        report.append("=" * 80)

        summary = results_df.groupby('interval_ms').agg({
            'prr_prr_percent': 'mean',
            'prr_packet_loss_percent': 'mean',
            'quality_rssi_mean': 'mean',
            'quality_rssi_std': 'mean'
        }).reset_index()

        report.append("\nOverall Performance by Interval (averaged across LoS/nLoS):")
        report.append("-" * 80)
        for _, row in summary.iterrows():
            report.append(f"\n{int(row['interval_ms'])} ms:")
            report.append(f"  PRR: {row['prr_prr_percent']:.2f}%")
            report.append(f"  Loss: {row['prr_packet_loss_percent']:.2f}%")
            report.append(f"  RSSI: {row['quality_rssi_mean']:.2f} dBm")
            report.append(f"  Stability: {row['quality_rssi_std']:.2f} dB")

        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)

        # Find best interval for different use cases
        best_prr = summary.loc[summary['prr_prr_percent'].idxmax()]
        best_stability = summary.loc[summary['quality_rssi_std'].idxmin()]

        report.append(f"\nFor Maximum Reliability: {int(best_prr['interval_ms'])} ms")
        report.append(f"  - Highest PRR: {best_prr['prr_prr_percent']:.2f}%")
        report.append(f"\nFor Best Signal Stability: {int(best_stability['interval_ms'])} ms")
        report.append(f"  - Lowest RSSI Std Dev: {best_stability['quality_rssi_std']:.2f} dB")
        report.append(f"\nFor Power Efficiency: 1000 ms (fewer transmissions)")
        report.append(f"  - Trade-off: Lower update rate, but conserves battery")

        report.append("\n" + "=" * 80)

        # Save report
        report_text = "\n".join(report)
        with open(output_dir / 'advertisement_interval_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {output_dir / 'advertisement_interval_report.txt'}")


def main():
    """Main execution function"""

    # Setup paths
    base_path = Path("/home/user/BLE-Pedestrians")
    dataset_path = base_path / "dataset" / "Advertisement Interval"
    output_dir = base_path / "analysis" / "A2_advertisement_interval_impact" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BLE ADVERTISEMENT INTERVAL IMPACT ANALYSIS")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = AdvertisementIntervalAnalyzer(dataset_path)

    # Run comprehensive analysis
    print("Analyzing all intervals and orientations...")
    results_df = analyzer.analyze_all_intervals()

    if results_df.empty:
        print("Error: No data loaded. Check dataset paths.")
        return

    # Save results
    results_df.to_csv(output_dir / 'interval_analysis_results.csv', index=False)
    print(f"\nSaved analysis results to {output_dir / 'interval_analysis_results.csv'}")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    analyzer.plot_packet_reception_analysis(results_df, output_dir)
    analyzer.plot_signal_quality_analysis(results_df, output_dir)
    print()

    # Generate report
    print("Generating analysis report...")
    analyzer.generate_report(results_df, output_dir)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
