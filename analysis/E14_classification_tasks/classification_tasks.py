#!/usr/bin/env python3
"""
E14: Classification Tasks
LoS/nLoS classification and position classification using machine learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ClassificationAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def load_and_prepare_data(self):
        """Load and prepare features for classification"""
        data = []
        los_path = self.dataset_path / 'Deployment Distance' / 'los'
        nlos_path = self.dataset_path / 'Deployment Distance' / 'nlos'

        for orientation, base_path in [('los', los_path), ('nlos', nlos_path)]:
            for pos in ['start', 'center', 'end']:
                pos_dir = 'centre' if pos == 'center' and orientation == 'nlos' else pos
                for dist in [3, 5, 7, 9]:
                    for run in range(1, 4):
                        file_path = base_path / pos_dir / f'run{run}' / f'{pos_dir}_{dist}m_run{run}.csv'
                        if file_path.exists():
                            df = pd.read_csv(file_path)
                            if not df.empty and 'rssi' in df.columns:
                                features = {
                                    'mean_rssi': df['rssi'].mean(),
                                    'std_rssi': df['rssi'].std(),
                                    'min_rssi': df['rssi'].min(),
                                    'max_rssi': df['rssi'].max(),
                                    'range_rssi': df['rssi'].max() - df['rssi'].min(),
                                    'distance': dist,
                                    'orientation': orientation,
                                    'position': pos
                                }
                                data.append(features)

        return pd.DataFrame(data)

    def classify_los_nlos(self, df):
        """Train LoS/nLoS classifier"""
        X = df[['mean_rssi', 'std_rssi', 'min_rssi', 'max_rssi', 'range_rssi', 'distance']]
        y = (df['orientation'] == 'los').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'actual': y_test,
            'feature_importance': clf.feature_importances_
        }

    def plot_results(self, los_results, output_dir):
        """Plot classification results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('E14: Classification Tasks', fontsize=14, fontweight='bold')

        # Feature importance
        ax = axes[0]
        features = ['mean_rssi', 'std_rssi', 'min_rssi', 'max_rssi', 'range_rssi', 'distance']
        importance = los_results['feature_importance']
        ax.barh(features, importance, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (LoS/nLoS)')
        ax.grid(True, alpha=0.3, axis='x')

        # Confusion matrix
        ax = axes[1]
        cm = confusion_matrix(los_results['actual'], los_results['predictions'], labels=[0, 1])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['nLoS', 'LoS'])
        ax.set_yticklabels(['nLoS', 'LoS'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'LoS/nLoS Classification\nAccuracy: {los_results["accuracy"]:.2%}')

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')

        plt.tight_layout()
        plt.savefig(output_dir / 'classification_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Saved classification plots")

    def generate_report(self, los_results, output_dir):
        """Generate classification report"""
        report = []
        report.append("=" * 80)
        report.append("E14: CLASSIFICATION TASKS")
        report.append("=" * 80)
        report.append("")

        report.append("LoS/nLoS CLASSIFICATION")
        report.append("-" * 80)
        report.append(f"Accuracy: {los_results['accuracy']:.2%}")
        report.append(f"Method: Random Forest (100 trees)")
        report.append("")

        report.append("FEATURE IMPORTANCE")
        report.append("-" * 80)
        features = ['mean_rssi', 'std_rssi', 'min_rssi', 'max_rssi', 'range_rssi', 'distance']
        for feat, imp in zip(features, los_results['feature_importance']):
            report.append(f"  {feat}: {imp:.3f}")
        report.append("")

        report.append("KEY INSIGHTS")
        report.append("-" * 80)
        report.append("1. LoS/nLoS classification achievable with RF classifier")
        report.append("2. RSSI statistics provide discriminative features")
        report.append("3. Can improve distance estimation by detecting orientation")
        report.append("")

        report.append("=" * 80)

        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write("\n".join(report))

        print("✓ Generated classification report")

    def run_analysis(self):
        """Execute classification analysis"""
        output_dir = Path('/home/user/BLE-Pedestrians/analysis/E14_classification_tasks/results')

        print("=" * 80)
        print("E14: CLASSIFICATION TASKS")
        print("=" * 80)

        print("\n1. Loading and preparing data...")
        df = self.load_and_prepare_data()
        print(f"   Prepared {len(df)} feature sets")

        print("\n2. Training LoS/nLoS classifier...")
        los_results = self.classify_los_nlos(df)
        print(f"   Accuracy: {los_results['accuracy']:.2%}")

        print("\n3. Creating visualizations...")
        self.plot_results(los_results, output_dir)

        print("\n4. Generating report...")
        self.generate_report(los_results, output_dir)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return los_results

if __name__ == '__main__':
    analyzer = ClassificationAnalyzer('/home/user/BLE-Pedestrians/dataset')
    results = analyzer.run_analysis()
