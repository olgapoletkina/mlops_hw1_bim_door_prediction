"""
Model Training Script
Trains CatBoost model using leave-one-project-out validation
"""

import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def convert_prediction_to_real_coordinates(predicted_fraction, wall_start_coord, wall_end_coord):
    """Convert predicted fraction (0-1) to real coordinates"""
    return wall_start_coord + predicted_fraction * (wall_end_coord - wall_start_coord)


def calculate_spatial_errors(comparison_df):
    """Calculate spatial errors in real coordinates"""
    errors = []
    
    for idx, row in comparison_df.iterrows():
        wall_axis = row['wall_long_side_axis']
        
        if wall_axis == 'x':
            actual_x = row['actual_real_coord']
            predicted_x = row['predicted_real_coord']
            actual_y = row['door_center_y']
            predicted_y = row['door_center_y']
        else:  # y
            actual_x = row['door_center_x']
            predicted_x = row['door_center_x']
            actual_y = row['actual_real_coord']
            predicted_y = row['predicted_real_coord']
        
        # Euclidean distance
        error = np.sqrt((actual_x - predicted_x)**2 + (actual_y - predicted_y)**2)
        errors.append(error)
    
    return np.array(errors)


def plot_predictions(comparison_df, test_project_id, metrics):
    """Plot prediction results"""
    # Add actual and predicted X,Y coordinates
    comparison_df['actual_x_coord'] = np.nan
    comparison_df['actual_y_coord'] = np.nan
    comparison_df['predicted_x_coord'] = np.nan
    comparison_df['predicted_y_coord'] = np.nan
    
    for idx, row in comparison_df.iterrows():
        wall_axis = row['wall_long_side_axis']
        
        if wall_axis == 'x':
            comparison_df.loc[idx, 'actual_x_coord'] = row['actual_real_coord']
            comparison_df.loc[idx, 'predicted_x_coord'] = row['predicted_real_coord']
            comparison_df.loc[idx, 'actual_y_coord'] = row['door_center_y']
            comparison_df.loc[idx, 'predicted_y_coord'] = row['door_center_y']
        else:  # y
            comparison_df.loc[idx, 'actual_y_coord'] = row['actual_real_coord']
            comparison_df.loc[idx, 'predicted_y_coord'] = row['predicted_real_coord']
            comparison_df.loc[idx, 'actual_x_coord'] = row['door_center_x']
            comparison_df.loc[idx, 'predicted_x_coord'] = row['door_center_x']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Spatial plot with error vectors
    ax = axes[0, 0]
    ax.quiver(
        comparison_df["actual_x_coord"],
        comparison_df["actual_y_coord"],
        comparison_df["predicted_x_coord"] - comparison_df["actual_x_coord"],
        comparison_df["predicted_y_coord"] - comparison_df["actual_y_coord"],
        angles="xy", scale_units="xy", scale=1,
        color="red", alpha=0.5, width=0.003, headwidth=4, headlength=6
    )
    
    sc = ax.scatter(
        comparison_df["actual_x_coord"],
        comparison_df["actual_y_coord"],
        c=comparison_df["error_real_coord"],
        cmap="coolwarm",
        alpha=0.9, s=80
    )
    
    arrow_proxy = mpatches.FancyArrow(0, 0, 0.3, 0, color="red", width=0.02,
                                      head_width=0.01, head_length=0.15)
    ax.legend([sc, arrow_proxy], ["Actual Doors", "Error Vector"])
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Prediction Errors - Test Project {test_project_id}")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Error (distance)")
    
    # 2. Error histogram
    ax = axes[0, 1]
    ax.hist(comparison_df["error_real_coord"], bins=50, color="orange", alpha=0.7, edgecolor='black')
    ax.axvline(metrics['median_error'], color='red', linestyle='--', linewidth=2, 
               label=f'Median: {metrics["median_error"]:.3f}')
    ax.axvline(metrics['mean_error'], color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {metrics["mean_error"]:.3f}')
    ax.set_xlabel("Error (distance)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Prediction Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Parity plot
    ax = axes[1, 0]
    ax.scatter(comparison_df["actual_fraction"], comparison_df["predicted_fraction"], 
               alpha=0.6, s=50)
    lims = [0, 1]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel("Actual Position (fraction)")
    ax.set_ylabel("Predicted Position (fraction)")
    ax.set_title("Parity Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 4. Metrics text
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_text = f"""
    Test Project: {test_project_id}
    Number of Doors: {metrics['n_doors']}
    
    Error Metrics:
    ─────────────────────────
    Median Error: {metrics['median_error']:.4f}
    Mean Error: {metrics['mean_error']:.4f}
    MAE: {metrics['mae']:.4f}
    RMSE: {metrics['rmse']:.4f}
    Max Error: {metrics['max_error']:.4f}
    
    Percentiles:
    ─────────────────────────
    25th: {metrics['p25']:.4f}
    50th: {metrics['p50']:.4f}
    75th: {metrics['p75']:.4f}
    90th: {metrics['p90']:.4f}
    95th: {metrics['p95']:.4f}
    """
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def main():
    """Main training pipeline with leave-one-project-out validation"""
    print("=" * 80)
    print("MODEL TRAINING PIPELINE (Leave-One-Project-Out)")
    print("=" * 80)
    
    # Load parameters
    params = load_params()
    processed_path = params['data']['processed_path']
    project_ids = params['data']['project_ids']
    test_project_id = params['train']['test_project_id']
    feature_cols = params['features']['feature_columns']
    target_col = params['features']['target_column']
    catboost_params = params['model']['catboost']
    
    # Setup MLflow
    mlflow_tracking_uri = params['mlflow']['tracking_uri']
    mlflow_experiment = params['mlflow']['experiment_name']
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    
    print(f"\nTest Project: {test_project_id}")
    print(f"Train Projects: {[p for p in project_ids if p != test_project_id]}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    
    # Load data
    print("\n[1/4] Loading processed data...")
    train_dfs = []
    for proj_id in project_ids:
        if proj_id != test_project_id:
            df = pd.read_csv(f"{processed_path}/project_{proj_id}.csv")
            train_dfs.append(df)
            print(f"  Train: Project {proj_id} - {len(df)} doors")
    
    test_df = pd.read_csv(f"{processed_path}/project_{test_project_id}.csv")
    print(f"  Test:  Project {test_project_id} - {len(test_df)} doors")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    print(f"\nTotal train samples: {len(train_df)}")
    print(f"Total test samples: {len(test_df)}")
    
    # Prepare features and target
    print("\n[2/4] Preparing features...")
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    print(f"  Features: {feature_cols}")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    
    # Start MLflow run
    with mlflow.start_run():
        
        # Train model
        print("\n[3/4] Training CatBoost model...")
        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            verbose=False
        )
        print("  ✓ Model trained")
        
        # Make predictions
        print("\n[4/4] Evaluating model...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Create comparison dataframe with spatial information
        comparison_df = test_df[['door_id', 'wall_long_start_coord', 'wall_long_end_coord',
                                 'wall_long_side_axis', 'door_center_x', 'door_center_y']].copy()
        comparison_df['actual_fraction'] = y_test.values
        comparison_df['predicted_fraction'] = y_pred_test
        
        # Convert to real coordinates
        comparison_df['actual_real_coord'] = convert_prediction_to_real_coordinates(
            comparison_df['actual_fraction'],
            comparison_df['wall_long_start_coord'],
            comparison_df['wall_long_end_coord']
        )
        
        comparison_df['predicted_real_coord'] = convert_prediction_to_real_coordinates(
            comparison_df['predicted_fraction'],
            comparison_df['wall_long_start_coord'],
            comparison_df['wall_long_end_coord']
        )
        
        # Calculate errors
        spatial_errors = calculate_spatial_errors(comparison_df)
        comparison_df['error_real_coord'] = spatial_errors
        comparison_df['error_fraction'] = abs(comparison_df['actual_fraction'] - 
                                             comparison_df['predicted_fraction'])
        
        # Calculate all metrics
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test),
            'median_error': np.median(spatial_errors),
            'mean_error': np.mean(spatial_errors),
            'max_error': np.max(spatial_errors),
            'p25': np.percentile(spatial_errors, 25),
            'p50': np.percentile(spatial_errors, 50),
            'p75': np.percentile(spatial_errors, 75),
            'p90': np.percentile(spatial_errors, 90),
            'p95': np.percentile(spatial_errors, 95),
            'n_doors': len(comparison_df)
        }
        
        # Print metrics
        print("\n" + "=" * 80)
        print("METRICS")
        print("=" * 80)
        
        print("\nTrain Metrics (Fraction-based):")
        print(f"  MAE:  {train_metrics['mae']:.6f}")
        print(f"  RMSE: {train_metrics['rmse']:.6f}")
        print(f"  R2:   {train_metrics['r2']:.6f}")
        
        print("\nTest Metrics (Fraction-based):")
        print(f"  MAE:  {test_metrics['mae']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  R2:   {test_metrics['r2']:.6f}")
        
        print("\nTest Metrics (Spatial - Real Coordinates):")
        print(f"  Median Error: {test_metrics['median_error']:.4f}")
        print(f"  Mean Error:   {test_metrics['mean_error']:.4f}")
        print(f"  Max Error:    {test_metrics['max_error']:.4f}")
        
        print("\nError Percentiles:")
        print(f"  25th: {test_metrics['p25']:.4f}")
        print(f"  50th: {test_metrics['p50']:.4f}")
        print(f"  75th: {test_metrics['p75']:.4f}")
        print(f"  90th: {test_metrics['p90']:.4f}")
        print(f"  95th: {test_metrics['p95']:.4f}")
        print("=" * 80)
        
        # Log parameters to MLflow
        mlflow.log_param("test_project_id", test_project_id)
        mlflow.log_param("n_train_projects", len(project_ids) - 1)
        mlflow.log_params(catboost_params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log metrics to MLflow
        mlflow.log_metric("train_mae", train_metrics['mae'])
        mlflow.log_metric("train_rmse", train_metrics['rmse'])
        mlflow.log_metric("train_r2", train_metrics['r2'])
        mlflow.log_metric("test_mae", test_metrics['mae'])
        mlflow.log_metric("test_rmse", test_metrics['rmse'])
        mlflow.log_metric("test_r2", test_metrics['r2'])
        
        # Log spatial metrics
        mlflow.log_metric("test_median_error", test_metrics['median_error'])
        mlflow.log_metric("test_mean_error", test_metrics['mean_error'])
        mlflow.log_metric("test_max_error", test_metrics['max_error'])
        mlflow.log_metric("test_p25", test_metrics['p25'])
        mlflow.log_metric("test_p50", test_metrics['p50'])
        mlflow.log_metric("test_p75", test_metrics['p75'])
        mlflow.log_metric("test_p90", test_metrics['p90'])
        mlflow.log_metric("test_p95", test_metrics['p95'])
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.to_string(index=False))
        
        # Create and save visualization
        print("\nGenerating visualization...")
        fig = plot_predictions(comparison_df, test_project_id, test_metrics)
        
        plot_path = f"models/prediction_plot_project_{test_project_id}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Visualization saved: {plot_path}")
        
        # Save artifacts
        Path("models").mkdir(exist_ok=True)
        
        # Save feature importance
        feature_importance_path = "models/feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path)
        
        # Save comparison results
        comparison_path = f"models/predictions_project_{test_project_id}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        mlflow.log_artifact(comparison_path)
        
        # Log visualization
        mlflow.log_artifact(plot_path)
        
        # Set run name and tags
        mlflow.set_tag("mlflow.runName", f"test_project_{test_project_id}")
        mlflow.set_tag("test_project", test_project_id)
        mlflow.set_tag("model_type", "CatBoost")
        
        # Save model
        model_path = f"models/catboost_model_test_{test_project_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        
        # Log model to MLflow
        mlflow.catboost.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        print("\n✓ All artifacts logged to MLflow")
        print(f"\nTo view results, run: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
        print("=" * 80)


if __name__ == "__main__":
    main()