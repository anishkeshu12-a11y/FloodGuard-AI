import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("="*50)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    # Load dataset
    data_path = 'data/flood_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run src/data_generator.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nDataset Info:")
    print(df.info())
    
    # Create output directory for figures
    figures_dir = os.path.join('reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\nFigures will be saved in: {figures_dir}")
    
    # 1. Missing Value Analysis
    print("\n--- 1. Missing Value Analysis ---")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Counts': missing_counts, 'Percentage (%)': missing_percentages})
    print(missing_df)
    
    # Plot missing values
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_df.index, y='Percentage (%)', data=missing_df, palette='viridis')
    plt.title('Percentage of Missing Values per Feature', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'missing_values.png'), dpi=300)
    plt.close()
    print("Saved: missing_values.png")
    
    # 2. Outlier Detection (using IQR)
    print("\n--- 2. Outlier Detection (IQR Method) ---")
    numeric_features = ['elevation', 'slope', 'rainfall', 'rainfall_lag_1', 'rainfall_lag_2', 'temperature', 'humidity']
    
    outlier_summary = {}
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df)) * 100
        outlier_summary[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outlier Count': len(outliers),
            'Outlier Percentage (%)': round(outlier_percentage, 2)
        }
        print(f"Feature '{col}': found {len(outliers)} outliers ({outlier_percentage:.2f}%)")
        
    # Plot boxplot for outliers
    plt.figure(figsize=(12, 6))
    # Standardize data briefly just for visualization comparisons in boxplots
    df_normalized = (df[numeric_features] - df[numeric_features].mean()) / df[numeric_features].std()
    sns.boxplot(data=df_normalized, palette='Set2')
    plt.title('Outlier Detection (Standardized Features Boxplot)', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Standardized Value', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'outliers_boxplot.png'), dpi=300)
    plt.close()
    print("Saved: outliers_boxplot.png")
    
    # 3. Correlation Heatmap
    print("\n--- 3. Generating Correlation Heatmap ---")
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print("Saved: correlation_heatmap.png")
    
    # 4. Feature Distribution Plots
    print("\n--- 4. Plotting Feature Distributions ---")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_features):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='skyblue', stat='density')
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Density')
        
    # Deactivate the unused subplots
    for i in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[i])
        
    plt.suptitle('Continuous Features Probability Distribution Curves', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(figures_dir, 'feature_distributions.png'), dpi=300)
    plt.close()
    print("Saved: feature_distributions.png")
    
    # 5. Class Imbalance Visualization
    print("\n--- 5. Class Imbalance Visualization ---")
    plt.figure(figsize=(8, 5))
    # Map labels for nicer display
    risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    risk_counts = df['flood_risk'].value_counts().rename(index=risk_labels)
    
    sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=['green', 'orange', 'red'])
    plt.title('Target variable distribution: Flood Risk Classes', fontsize=14, fontweight='bold')
    plt.xlabel('Risk Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add counts on top of bars
    for index, value in enumerate(risk_counts.values):
        plt.text(index, value + 100, str(value), ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'class_imbalance.png'), dpi=300)
    plt.close()
    print("Saved: class_imbalance.png")
    
    # 6. Rainfall Trend Analysis
    print("\n--- 6. Rainfall Trend Analysis ---")
    daily_rainfall = df.groupby('day')['rainfall'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.plot(daily_rainfall['day'], daily_rainfall['rainfall'], marker='o', color='royalblue', linewidth=2, markersize=6)
    plt.title('Daily Average Rainfall Trend Over Simulation Period', fontsize=14, fontweight='bold')
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Average Rainfall (mm)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'rainfall_trend.png'), dpi=300)
    plt.close()
    print("Saved: rainfall_trend.png")
    
    print("\n" + "="*50)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    run_eda()
