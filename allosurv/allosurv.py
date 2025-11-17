import os, warnings, math, json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, precision_recall_curve,
    roc_curve
)
from sklearn.calibration import calibration_curve
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import shap
shap.initjs()
warnings.filterwarnings('ignore')

def make_preprocessor(X):
    '''
    Create a preprocessing pipeline for numerical and categorical features.
    
    Parameters:
    X : pd.DataFrame
        Input feature dataframe.
    Returns:
    pre : ColumnTransformer
        Preprocessing pipeline.
    
    '''
    num_cols = [c for c in X.columns if X[c].dtype != 'object']
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=False)), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent', add_indicator=False)), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ], remainder='drop', sparse_threshold=0.0)    
    return pre

def make_models_and_grids():
    '''
    Create a dictionary of models and their hyperparameter grids.
    Returns:
    models : dict
        Dictionary of model name to model instance.
    grids : dict
        Dictionary of model name to hyperparameter grid.
    '''
    models = {
        'LogReg_EN': LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=5000, random_state=42),
        'GradBoost': GradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced') # adding random forest model
    }
    grids = {
        'LogReg_EN': {'clf__C':[0.05,0.1,1.0,10.0], 'clf__l1_ratio':[0.0,0.5,1.0]},
        'GradBoost': {'clf__n_estimators':[150,300], 'clf__learning_rate':[0.05,0.1], 'clf__max_depth':[2,3], 'clf__min_samples_leaf':[1,5]},
        'RandomForest': {'clf__n_estimators':[100,200], 'clf__max_depth':[None,10,20], 'clf__min_samples_split':[2,5]} # hyperparameter grid for random forest
    }
    return models, grids

def make_labels(df, T):
    '''
    Create binary labels for survival analysis at time T.
    Parameters:
    df : pd.DataFrame
        Dataframe containing 'vital_status' and 'survival_days' columns.
    T : float
        Time threshold for survival.
    Returns:
    pd.Series
        Binary labels (1 if event occurred by time T, 0 if censored after T, NaN if censored before T).
    '''
    vs = df['vital_status'].values.astype(int)
    sd = df['survival_days'].values.astype(float)
    y = np.full(len(df), np.nan)
    y[(vs==1) & (sd<=T)] = 1.0
    y[(sd>T)] = 0.0
    y[(vs==1) & (sd>T)] = 0.0
    return pd.Series(y)

def evaluate(y_true, y_prob, y_pred):
    '''
    Evaluate model performance using various metrics.
    Parameters:
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    y_pred : array-like
        Predicted binary labels.
    Returns:
    dict
        Dictionary of evaluation metrics.
    '''
    out = {}
    out['AUROC'] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan
    out['AUPRC'] = average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan
    out['Brier'] = brier_score_loss(y_true, y_prob)
    out['Accuracy'] = accuracy_score(y_true, y_pred)
    out['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['F1'] = f1_score(y_true, y_pred, zero_division=0)
    out['BalancedAcc'] = balanced_accuracy_score(y_true, y_pred)
    return out

def plot_roc_pr_cal(y_true, y_prob, title_prefix):
    '''
    Plot ROC curve, Precision-Recall curve, and Calibration curve.
    Parameters:
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    title_prefix : str
        Prefix for plot titles.
    '''
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(); plt.plot(fpr,tpr,label='ROC'); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{title_prefix} - ROC'); plt.legend(); plt.tight_layout(); plt.show()
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rec,prec,label='PR'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{title_prefix} - PR'); plt.legend(); plt.tight_layout(); plt.show()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    plt.figure(); plt.plot(prob_pred, prob_true, 'o-', label='Calibration'); plt.plot([0,1],[0,1],'--'); plt.xlabel('Predicted'); plt.ylabel('Observed'); plt.title(f'{title_prefix} - Calibration'); plt.legend(); plt.tight_layout(); plt.show()



def plot_shap_for_horizon(best_models, horizon, Xh, max_display=20, plot_type='summary', 
                          clean_names=True, figsize=(10, 6), save_path=None):
    '''
    Plot SHAP analysis for the best model at a given time horizon.

    Parameters:
    -----------
    best_models : dict or str
        Dictionary mapping time horizon to model file path, or direct path to model file.
    horizon : str
        Time horizon key (e.g., '3y', '1y', '100d'). Ignored if best_models is a string path.
    Xh : pd.DataFrame
        Feature dataframe for SHAP analysis.
    max_display : int, default=20
        Maximum number of features to display in the plot.
    plot_type : str, default='summary'
        Type of SHAP plot: 'summary', 'bar', 'waterfall', or 'force'.
    clean_names : bool, default=True
        Whether to clean feature names (remove prefixes, convert to uppercase).
    figsize : tuple, default=(10, 6)
        Figure size as (width, height).
    save_path : str or None, default=None
        Path to save the plot. If None, plot is shown but not saved.

    Returns:
    --------
    dict
        Dictionary containing SHAP values, feature names, and other analysis results.

    Examples:
    ---------
    # Basic usage
    plot_shap_for_horizon(best_models, '3y', Xh)
    
    # With customization
    plot_shap_for_horizon(best_models, '1y', Xh, max_display=15, plot_type='bar')
    
    # Save plot
    plot_shap_for_horizon(best_models, '2y', Xh, save_path='shap_2y.png')
    
    # Use direct model path
    plot_shap_for_horizon('path/to/model.joblib', None, Xh)
    '''
    try:
        # Handle both dict and direct path inputs
        if isinstance(best_models, dict):
            if horizon not in best_models:
                available = list(best_models.keys())
                raise ValueError(f"Horizon '{horizon}' not found. Available horizons: {available}")
            model_path = best_models[horizon]
            title_horizon = horizon.upper()
        else:
            model_path = best_models
            title_horizon = horizon.upper() if horizon else "MODEL"
        
        print(f"Loading model for {title_horizon} horizon...")
        best_model = joblib.load(model_path)
        
        # Extract pipeline and transform data
        pipeline = best_model.estimator
        print("Transforming features...")
        X_transformed = pipeline.named_steps['prep'].transform(Xh)
        
        # Get the underlying classifier
        clf = pipeline.named_steps['clf']
        print(f"Creating SHAP explainer for {type(clf).__name__}...")
        
        # Choose appropriate explainer
        if isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.Explainer(clf, X_transformed)

        print("Computing SHAP values...")
        shap_values = explainer(X_transformed)
        
        # Get feature names
        feature_names = pipeline.named_steps['prep'].get_feature_names_out()
        
        if clean_names:
            clean_feature_names = [name.replace('num__','').replace('cat__oh__','') for name in feature_names]
            display_names = [name.upper() for name in clean_feature_names]
        else:
            display_names = list(feature_names)
        
        # Handle binary classification SHAP values
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_values_plot = shap_values.values[:, :, 1]  # Class 1 (positive class)
            print("Using SHAP values for positive class (class 1)")
        else:
            shap_values_plot = shap_values.values
            print("Using raw SHAP values")
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if plot_type == 'summary':
            shap.summary_plot(shap_values_plot, X_transformed, feature_names=display_names, 
                            show=False, max_display=max_display)
            plt.title(f'SHAP Summary Plot - {title_horizon} Horizon')
            
        elif plot_type == 'bar':
            shap.summary_plot(shap_values_plot, X_transformed, feature_names=display_names, 
                            plot_type='bar', show=False, max_display=max_display)
            plt.title(f'SHAP Feature Importance - {title_horizon} Horizon')
            
        elif plot_type == 'waterfall':
            if len(shap_values_plot) > 0:
                shap.plots.waterfall(shap.Explanation(values=shap_values_plot[0], 
                                                    base_values=explainer.expected_value, 
                                                    feature_names=display_names), 
                                    max_display=max_display, show=False)
                plt.title(f'SHAP Waterfall Plot - {title_horizon} Horizon (Sample 0)')
            else:
                print("No samples available for waterfall plot")
                return None
                
        elif plot_type == 'force':
            if len(shap_values_plot) > 0:
                shap.force_plot(explainer.expected_value, shap_values_plot[0], 
                              X_transformed[0], feature_names=display_names, 
                              matplotlib=True, show=False)
                plt.title(f'SHAP Force Plot - {title_horizon} Horizon (Sample 0)')
            else:
                print("No samples available for force plot")
                return None
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Choose from: 'summary', 'bar', 'waterfall', 'force'")
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Return useful information
        result = {
            'shap_values': shap_values_plot,
            'feature_names': display_names,
            'X_transformed': X_transformed,
            'explainer': explainer,
            'model_type': type(clf).__name__,
            'n_features': len(display_names),
            'n_samples': len(X_transformed)
        }
        
        print(f"✓ SHAP analysis complete for {title_horizon} horizon")
        print(f"  Model type: {result['model_type']}")
        print(f"  Features: {result['n_features']}, Samples: {result['n_samples']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        raise e


def plot_shap_all_horizons(best_models, Xh, **kwargs):
    '''
    Plot SHAP analysis for all available horizons.
    
    Parameters:
    -----------
    best_models : dict
        Dictionary mapping time horizon to model file path.
    Xh : pd.DataFrame
        Feature dataframe for SHAP analysis.
    **kwargs : additional arguments
        Additional arguments passed to plot_shap_for_horizon.
    
    Returns:
    --------
    dict
        Dictionary of results for each horizon.
    '''
    results = {}
    
    print(f"Plotting SHAP for {len(best_models)} horizons...")
    for i, horizon in enumerate(best_models.keys(), 1):
        print(f"\n[{i}/{len(best_models)}] Processing {horizon} horizon...")
        try:
            results[horizon] = plot_shap_for_horizon(best_models, horizon, Xh, **kwargs)
        except Exception as e:
            print(f"❌ Failed for {horizon}: {str(e)}")
            results[horizon] = None
    
    success_count = sum(1 for r in results.values() if r is not None)
    print(f"\n✓ Completed: {success_count}/{len(best_models)} horizons processed successfully")
    
    return results

def plot_performance_curves(curve_data_list, figsize=(15, 5), save_prefix=None, separate_plots=False, 
                           show_full_curves=False, pr_log_scale=False):
    '''
    Plot superimposed ROC, PR, and Calibration curves from multiple models/horizons.
    
    Parameters:
    -----------
    curve_data_list : list of tuples
        List of (y_true, y_prob, label) tuples for each model/horizon.
        Same data you would pass to plot_roc_pr_cal individually.
    figsize : tuple, default=(15, 5) for subplots, (10, 6) for separate
        Figure size as (width, height).
    save_prefix : str or None, default=None
        Prefix for saving plots.
    separate_plots : bool, default=False
        If True, create three separate plots. If False, create subplots.
    show_full_curves : bool, default=False
        If True, adjust y-axis to show complete curves including low values.
    pr_log_scale : bool, default=False
        If True, use log scale for PR curve y-axis to better show low precision values.
    
    Usage:
    ------
    # Collect data the same way you use plot_roc_pr_cal
    curve_data = [
        (y_true_1y, y_prob_1y, '1y (RandomForest)'),
        (y_true_2y, y_prob_2y, '2y (GradBoost)'),
        (y_true_3y, y_prob_3y, '3y (LogReg)')
    ]
    
    # Combined subplots (default)
    plot_performance_curves(curve_data)
    
    # Separate plots
    plot_performance_curves(curve_data, separate_plots=True)
    '''
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    if separate_plots:
        # Adjust default figsize for separate plots
        if figsize == (15, 5):  # Default subplot size
            figsize = (10, 6)
            
        # Plot 1: ROC Curves
        plt.figure(figsize=figsize)
        for i, (y_true, y_prob, label) in enumerate(curve_data_list):
            color = colors[i % len(colors)]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, color=color, linewidth=2, 
                    label=f'{label} (AUC={auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.despine()
        if save_prefix:
            plt.savefig(f'{save_prefix}_roc.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Precision-Recall Curves
        plt.figure(figsize=figsize)
        for i, (y_true, y_prob, label) in enumerate(curve_data_list):
            color = colors[i % len(colors)]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            auprc_score = average_precision_score(y_true, y_prob)
            plt.plot(rec, prec, color=color, linewidth=2,
                    label=f'{label} (AUPRC={auprc_score:.3f})')
        
        plt.xlim([0.0, 1.0])
        if show_full_curves:
            plt.ylim([-0.05, 1.05])
        else:
            plt.ylim([0.0, 1.05])
        if pr_log_scale:
            plt.yscale('log')
            plt.ylim([0.01, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.despine()
        if save_prefix:
            plt.savefig(f'{save_prefix}_pr.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Calibration Curves
        plt.figure(figsize=figsize)
        for i, (y_true, y_prob, label) in enumerate(curve_data_list):
            color = colors[i % len(colors)]
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
            plt.plot(prob_pred, prob_true, 'o-', color=color, 
                    linewidth=2, markersize=6, label=label)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.despine()
        if save_prefix:
            plt.savefig(f'{save_prefix}_calibration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    else:
        # Combined subplots (original behavior)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for i, (y_true, y_prob, label) in enumerate(curve_data_list):
            color = colors[i % len(colors)]
            
            # Compute curves (same as your individual function)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
            
            # Compute metrics for legend
            auc_score = roc_auc_score(y_true, y_prob)
            auprc_score = average_precision_score(y_true, y_prob)
            
            # Plot 1: ROC
            axes[0].plot(fpr, tpr, color=color, linewidth=2, 
                        label=f'{label} (AUC={auc_score:.3f})')
            
            # Plot 2: PR
            axes[1].plot(rec, prec, color=color, linewidth=2,
                        label=f'{label} (AUPRC={auprc_score:.3f})')
            
            # Plot 3: Calibration
            axes[2].plot(prob_pred, prob_true, 'o-', color=color, 
                        linewidth=2, markersize=6, label=label)
        
        # Style ROC plot
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlim([0.0, 1.0]); axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
        axes[0].set_title('ROC Curves - All Models')
        axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)
        
        # Style PR plot
        axes[1].set_xlim([0.0, 1.0])
        if show_full_curves:
            axes[1].set_ylim([-0.05, 1.05])
        else:
            axes[1].set_ylim([0.0, 1.05])
        if pr_log_scale:
            axes[1].set_yscale('log')
            axes[1].set_ylim([0.01, 1.05])
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
        axes[1].set_title('PR Curves - All Models')
        axes[1].legend(loc="lower left"); axes[1].grid(True, alpha=0.3)
        
        # Style Calibration plot
        axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[2].set_xlim([0.0, 1.0]); axes[2].set_ylim([0.0, 1.0])
        axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Observed')
        axes[2].set_title('Calibration Curves - All Models')
        axes[2].legend(loc="lower right"); axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f'{save_prefix}_combined.png', dpi=300, bbox_inches='tight')
        plt.show()