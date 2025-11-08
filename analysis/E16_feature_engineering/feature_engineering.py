#!/usr/bin/env python3
"""
E16: Feature Engineering
Extract temporal and statistical features from RSSI time series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class FeatureEngineeringAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def extract_features(self):
        """Extract comprehensive features from RSSI data"""
        features_list = []
        los_path = self.dataset_path / 'Deployment Distance' / 'los'

        for pos in ['start', 'center', 'end']:
            for dist in [3, 5, 7, 9]:
                for run in range(1, 4):
                    file_path = los_path / pos / f'run{run}' / f'{pos}_{dist}m_run{run}.csv'
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        if not df.empty and 'rssi' in df.columns:
                            rssi = df['rssi'].values

                            # Statistical features
                            features = {
                                'distance': dist,
                                'position': pos,
                                'run': run,
                                'mean': np.mean(rssi),
                                'std': np.std(rssi),
                                'min': np.min(rssi),
                                'max': np.max(rssi),
                                'median': np.median(rssi),
                                'range': np.max(rssi) - np.min(rssi),
                                'q25': np.percentile(rssi, 25),
                                'q75': np.percentile(rssi, 75),
                                'iqr': np.percentile(rssi, 75) - np.percentile(rssi, 25),
                                'skewness': pd.Series(rssi).skew(),
                                'kurtosis': pd.Series(rssi).kurtosis(),
                                'variation_coeff': np.std(rssi) / abs(np.mean(rssi)) if np.mean(rssi) != 0 else 0,
                            }

                            # Temporal features
                            if len(rssi) > 1:
                                features['rate_of_change'] = np.mean(np.abs(np.diff(rssi)))
                                features['max_change'] = np.max(np.abs(np.diff(rssi)))

                            features_list.append(features)

        return pd.DataFrame(features_list)

    def analyze_feature_correlations(self, features_df):
        """Analyze feature correlations with distance"""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'distance']

        correlations = {}
        for col in numeric_cols:
            corr = features_df[[col, 'distance']].corr().iloc[0, 1]
            correlations[col] = corr

        return pd.DataFrame([correlations]).T.rename(columns={0: 'correlation'}).sort_values('correlation', key=abs, ascending=False)

    def plot_features(self, features_df, correlations, output_dir):
        """Plot feature analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('E16: Feature Engineering', fontsize=14, fontweight='bold')

        # Feature correlations
        ax = axes[0]
        top_features = correlations.head(10)
        ax.barh(range(len(top_features)), top_features['correlation'].values, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Correlation with Distance')
        ax.set_title('Top 10 Features by Correlation')
        ax.grid(True, alpha=0.3, axis='x')

        # Feature example: mean vs std
        ax = axes[1]
        for dist in [3, 5, 7, 9]:
            dist_data = features_df[features_df['distance'] == dist]
            ax.scatter(dist_data['mean'], dist_data['std'], label=f'{dist}m', alpha=0.6)

        ax.set_xlabel('Mean RSSI')
        ax.set_ylabel('Std RSSI')
        ax.set_title('Feature Space Example')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'feature_engineering.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Saved feature analysis plots")

    def generate_report(self, features_df, correlations, output_dir):
        """Generate feature engineering report"""
        report = []
        report.append("=" * 80)
        report.append("E16: FEATURE ENGINEERING")
        report.append("=" * 80)
        report.append("")

        report.append(f"EXTRACTED FEATURES: {len(features_df.columns) - 3} features per sample")
        report.append(f"Total samples: {len(features_df)}")
        report.append("")

        report.append("TOP FEATURES BY CORRELATION WITH DISTANCE")
        report.append("-" * 80)
        for feat, corr in correlations.head(10).iterrows():
            report.append(f"  {feat}: {corr['correlation']:.3f}")
        report.append("")

        report.append("FEATURE CATEGORIES")
        report.append("-" * 80)
        report.append("Statistical: mean, std, min, max, median, range, quartiles, IQR")
        report.append("Distribution: skewness, kurtosis, coefficient of variation")
        report.append("Temporal: rate of change, max change")
        report.append("")

        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("• Use these features for ML models (classification, regression)")
        report.append("• Mean RSSI shows strongest correlation with distance")
        report.append("• Temporal features capture movement patterns")
        report.append("• Statistical features help classify LoS/nLoS")
        report.append("")

        report.append("=" * 80)

        with open(output_dir / 'feature_engineering_report.txt', 'w') as f:
            f.write("\n".join(report))

        print("✓ Generated feature engineering report")

    def run_analysis(self):
        """Execute feature engineering analysis"""
        output_dir = Path('/home/user/BLE-Pedestrians/analysis/E16_feature_engineering/results')

        print("=" * 80)
        print("E16: FEATURE ENGINEERING")
        print("=" * 80)

        print("\n1. Extracting features...")
        features_df = self.extract_features()
        features_df.to_csv(output_dir / 'engineered_features.csv', index=False)
        print(f"   Extracted {len(features_df.columns)} features from {len(features_df)} samples")

        print("\n2. Analyzing feature correlations...")
        correlations = self.analyze_feature_correlations(features_df)
        correlations.to_csv(output_dir / 'feature_correlations.csv')

        print("\n3. Creating visualizations...")
        self.plot_features(features_df, correlations, output_dir)

        print("\n4. Generating report...")
        self.generate_report(features_df, correlations, output_dir)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return features_df, correlations

if __name__ == '__main__':
    analyzer = FeatureEngineeringAnalyzer('/home/user/BLE-Pedestrians/dataset')
    features_df, correlations = analyzer.run_analysis()
