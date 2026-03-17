# === PREPROCESSING AND FEATURE EXTRACTION PIPELINE ===
# Splits dataset into train/test (80/20) and generates statistical features
# Features: 24 statistics per variable (Fx, Fy, Fz, Tx, Ty, Tz)

import os
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATION
# ====================================================================

# Split parameters
SPLIT_CONFIG = {
    'split_method': 'holdout',  # Options: 'holdout', 'none' (feature generation only)
    'test_size': 0.20,         # For holdout: % of test data
    'random_state': 42,
    'stratify': True           # Maintain class distribution
}

# OUTLIER TREATMENT CONFIGURATION ===
OUTLIER_CONFIG = {
    'enable_outlier_treatment': False,    # True/False - Enable outlier treatment
    'method': 'iqr',                     # Method: 'iqr', 'zscore', 'percentile', 'none'
    'iqr_multiplier': 1.5,               # IQR multiplier (1.5 = standard, 2.0 = more conservative)
    'zscore_threshold': 3.0,             # Z-score threshold (3.0 = standard)
    'percentile_lower': 2.5,             # Lower percentile for removal (2.5%)
    'percentile_upper': 97.5,            # Upper percentile for removal (97.5%)
    'min_samples_after_treatment': 3     # Minimum samples that must remain after treatment
}

# Output directory
OUTPUT_FOLDER = 'Preprocessed_Data'

print(f"\n{'='*60}")
print(" PREPROCESSING AND FEATURE EXTRACTION PIPELINE")
print(f"{'='*60}")
print(f" Configuration:")
print(f"   Split: {(1-SPLIT_CONFIG['test_size'])*100:.0f}% train / {SPLIT_CONFIG['test_size']*100:.0f}% test")
print(f"   Random state: {SPLIT_CONFIG['random_state']}")
print(f"   Stratified: {SPLIT_CONFIG['stratify']}")
print(f"   Output folder: {OUTPUT_FOLDER}")

print(f"\n Outlier Configuration:")
if OUTLIER_CONFIG['enable_outlier_treatment']:
    print(f"   Outlier treatment: ENABLED")
    print(f"   Method: {OUTLIER_CONFIG['method'].upper()}")
    
    if OUTLIER_CONFIG['method'] == 'iqr':
        print(f"   IQR Multiplier: {OUTLIER_CONFIG['iqr_multiplier']}")
    elif OUTLIER_CONFIG['method'] == 'zscore':
        print(f"   Z-Score Threshold: {OUTLIER_CONFIG['zscore_threshold']}")
    elif OUTLIER_CONFIG['method'] == 'percentile':
        print(f"   Percentiles: {OUTLIER_CONFIG['percentile_lower']}% - {OUTLIER_CONFIG['percentile_upper']}%")
    
    print(f"   Min samples after treatment: {OUTLIER_CONFIG['min_samples_after_treatment']}")
else:
    print(f"   Outlier treatment: DISABLED")
    print(f"   All samples will be preserved for feature calculation")

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def validate_data_integrity(X_data, y_data, stage_name, verbose=True):
    """Validates if X and y are aligned and without issues"""
    if verbose:
        print(f"\n VALIDATING: {stage_name}")
    
    issues = []
    
    if X_data.empty:
        issues.append("X_data is empty")
    if len(y_data) == 0:
        issues.append("y_data is empty")
    
    if len(X_data) != len(y_data):
        issues.append(f"Sizes don't align: X={len(X_data)}, y={len(y_data)}")
    
    if hasattr(y_data, 'isna'):
        missing_labels = y_data.isna().sum()
        if missing_labels > 0:
            issues.append(f"{missing_labels} missing labels")
    
    if not X_data.empty:
        total_nans = X_data.isna().sum().sum()
        if total_nans > 0 and verbose:
            print(f"   {total_nans} NaNs in features (normal if treated)")
    
    if issues:
        print(f"   ISSUES FOUND:")
        for issue in issues:
            print(f"      - {issue}")
        return False
    else:
        if verbose:
            print(f"   Data integrity: {len(X_data)} samples")
        return True

def safe_save_dataset(X_data, y_data, filepath, stage_name):
    """Saves dataset with complete integrity validation"""
    print(f"\n SAVING: {stage_name}")
    print(f"   File: {os.path.basename(filepath)}")
    
    if not validate_data_integrity(X_data, y_data, f"{stage_name} (pre-save)", verbose=False):
        print(f"  FAILURE: Invalid data for {stage_name}")
        return False
    
    try:
        X_aligned = X_data.copy()
        y_aligned = y_data.copy()
        
        # Reset indices
        if hasattr(X_aligned, 'reset_index'):
            X_aligned = X_aligned.reset_index(drop=True)
        if hasattr(y_aligned, 'reset_index'):
            y_aligned = y_aligned.reset_index(drop=True)
        elif hasattr(y_aligned, 'values'):
            y_aligned = pd.Series(y_aligned.values)
        else:
            y_aligned = pd.Series(y_aligned)
        
        # Check alignment after reset
        if len(X_aligned) != len(y_aligned):
            print(f"    ERROR: Sizes still don't align: X={len(X_aligned)}, y={len(y_aligned)}")
            return False
        
        # Remove rows with problematic labels
        if hasattr(y_aligned, 'isna'):
            valid_mask = y_aligned.notna() & (y_aligned != '') & (y_aligned != ' ')
            invalid_count = (~valid_mask).sum()
            
            if invalid_count > 0:
                print(f"    Removing {invalid_count} rows with invalid labels")
                X_aligned = X_aligned[valid_mask].reset_index(drop=True)
                y_aligned = y_aligned[valid_mask].reset_index(drop=True)
        
        # Check if any rows remain
        final_rows = len(X_aligned)
        if final_rows == 0:
            print(f"    ERROR: All rows were removed!")
            return False
        
        # Create final dataset
        dataset_final = X_aligned.copy()
        dataset_final['label'] = y_aligned
        
        # Final label verification
        label_check = dataset_final['label'].isna().sum()
        if label_check > 0:
            print(f"    ERROR: {label_check} NaN labels in final dataset!")
            return False
        
        # Save
        dataset_final.to_csv(filepath, index=False)
        
        # Verify saved file
        try:
            df_verify = pd.read_csv(filepath)
            verify_nan_labels = df_verify['label'].isna().sum()
            
            if verify_nan_labels > 0:
                print(f"    ERROR: {verify_nan_labels} NaN labels in saved file!")
                return False
            else:
                print(f"    SUCCESS: {df_verify.shape[0]} rows, {df_verify.shape[1]-1} features")
                print(f"    Classes: {df_verify['label'].unique().tolist()}")
                return True
                
        except Exception as e:
            print(f"    ERROR in verification: {e}")
            return False
            
    except Exception as e:
        print(f"    ERROR saving: {e}")
        return False

# ====================================================================
# OUTLIER TREATMENT FUNCTIONS
# ====================================================================

def apply_outlier_treatment(values, method='iqr', **kwargs):
    """
    Applies outlier treatment using different methods
    
    Parameters:
    -----------
    values : np.array
        Array of values for treatment
    method : str
        Treatment method: 'iqr', 'zscore', 'percentile', 'none'
    **kwargs : dict
        Method-specific parameters
        
    Returns:
    --------
    values_clean : np.array
        Array with outliers removed
    outliers_removed : int
        Number of outliers removed
    """
    
    if method == 'none' or len(values) == 0:
        return values, 0
    
    original_size = len(values)
    values_clean = values.copy()
    
    try:
        if method == 'iqr':
            # IQR (Interquartile Range) method
            multiplier = kwargs.get('iqr_multiplier', 2.0)
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            values_clean = values[(values >= lower_bound) & (values <= upper_bound)]
            
        elif method == 'zscore':
            # Z-Score method
            threshold = kwargs.get('zscore_threshold', 3.0)
            if len(values) > 1:
                z_scores = np.abs(stats.zscore(values))
                values_clean = values[z_scores < threshold]
            
        elif method == 'percentile':
            # Percentile method
            lower_percentile = kwargs.get('percentile_lower', 2.5)
            upper_percentile = kwargs.get('percentile_upper', 97.5)
            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)
            values_clean = values[(values >= lower_bound) & (values <= upper_bound)]
            
        else:
            print(f"    Unknown outlier method: {method}. Using original data.")
            return values, 0
        
        # Check if enough samples remain
        min_samples = kwargs.get('min_samples_after_treatment', 3)
        if len(values_clean) < min_samples:
            print(f"    Too few samples after treatment ({len(values_clean)}). Keeping original data.")
            return values, 0
            
        outliers_removed = original_size - len(values_clean)
        return values_clean, outliers_removed
        
    except Exception as e:
        print(f"    Error in outlier treatment: {e}. Using original data.")
        return values, 0

# ====================================================================
# STATISTICAL FEATURE EXTRACTION (MODIFIED)
# ====================================================================

def extract_statistical_features(row, variables, outlier_config=None):
    """
    Extracts 24 statistical features per variable:
    - 17 basic features: mean, std, max, min, median, iqr, skewness, kurtosis, 
                         rms, energy, power, zero_crossing_rate, first, last, 
                         peak_to_peak, mean_abs, entropy
    - 7 advanced features: variance, cv, p90, slope, crest_factor, shape_factor, peak_frequency
    
    MODIFIED: Includes flexible outlier treatment
    """
    
    # Default configuration if not provided
    if outlier_config is None:
        outlier_config = OUTLIER_CONFIG
    
    feats = {}
    total_outliers_removed = 0
    
    try:
        for var in variables:
            values = np.array(row[var])
            
            # Check if array is empty or all NaN
            if values.size == 0 or np.all(np.isnan(values)):
                # Default features for problematic cases
                default_features = {
                    f'{var}_mean': 0.0, f'{var}_std': 0.0, f'{var}_max': 0.0, f'{var}_min': 0.0,
                    f'{var}_median': 0.0, f'{var}_iqr': 0.0, f'{var}_skewness': 0.0, f'{var}_kurtosis': 0.0,
                    f'{var}_rms': 0.0, f'{var}_energy': 0.0, f'{var}_power': 0.0,
                    f'{var}_zero_crossing_rate': 0.0, f'{var}_first': 0.0, f'{var}_last': 0.0,
                    f'{var}_peak_to_peak': 0.0, f'{var}_mean_abs': 0.0, f'{var}_entropy': 0.0,
                    # Advanced features
                    f'{var}_variance': 0.0, f'{var}_cv': 0.0, f'{var}_p90': 0.0, f'{var}_slope': 0.0,
                    f'{var}_crest_factor': 0.0, f'{var}_shape_factor': 0.0, f'{var}_peak_frequency': 0.0
                }
                feats.update(default_features)
                continue
            
            # FLEXIBLE OUTLIER TREATMENT ===
            values_clean = values.copy()
            outliers_removed = 0
            
            if outlier_config['enable_outlier_treatment']:
                values_clean, outliers_removed = apply_outlier_treatment(
                    values,
                    method=outlier_config['method'],
                    iqr_multiplier=outlier_config.get('iqr_multiplier', 2.0),
                    zscore_threshold=outlier_config.get('zscore_threshold', 3.0),
                    percentile_lower=outlier_config.get('percentile_lower', 2.5),
                    percentile_upper=outlier_config.get('percentile_upper', 97.5),
                    min_samples_after_treatment=outlier_config.get('min_samples_after_treatment', 3)
                )
                total_outliers_removed += outliers_removed
            
            if len(values_clean) == 0:
                values_clean = values
            
            # Calculate basic statistics for percentiles
            if len(values_clean) > 1:
                Q1 = np.percentile(values_clean, 25)
                Q3 = np.percentile(values_clean, 75)
                IQR = Q3 - Q1
            else:
                Q1 = Q3 = IQR = values_clean[0] if len(values_clean) > 0 else 0.0
            
            # ============================================================
            # BASIC FEATURES (17)
            # ============================================================
            
            # Basic location and dispersion statistics
            feats[f'{var}_mean'] = np.mean(values_clean)
            feats[f'{var}_std'] = np.std(values_clean)
            feats[f'{var}_max'] = np.max(values) 
            feats[f'{var}_min'] = np.min(values)
            feats[f'{var}_median'] = np.median(values_clean)
            feats[f'{var}_iqr'] = IQR
            
            # Distribution shape statistics
            try:
                feats[f'{var}_skewness'] = stats.skew(values_clean) if len(values_clean) > 1 else 0.0
            except:
                feats[f'{var}_skewness'] = 0.0
                
            try:
                feats[f'{var}_kurtosis'] = stats.kurtosis(values_clean) if len(values_clean) > 1 else 0.0
            except:
                feats[f'{var}_kurtosis'] = 0.0
            
            # Energy and magnitude features
            feats[f'{var}_rms'] = np.sqrt(np.mean(values**2))
            feats[f'{var}_energy'] = np.sum(values**2)
            feats[f'{var}_power'] = np.mean(values**2)
            feats[f'{var}_mean_abs'] = np.mean(np.abs(values))
            
            # Temporal features
            feats[f'{var}_zero_crossing_rate'] = (np.diff(np.sign(values_clean)) != 0).sum() / (values_clean.size - 1) if values_clean.size > 1 else 0.0
            feats[f'{var}_first'] = values[0]   
            feats[f'{var}_last'] = values[-1]  
            feats[f'{var}_peak_to_peak'] = feats[f'{var}_max'] - feats[f'{var}_min']
            
            # Entropy (information)
            try:
                if values_clean.size > 1:
                    hist, _ = np.histogram(values_clean, bins='auto', density=True)
                    pk = hist[hist > 0]
                    if pk.size > 0:
                        feats[f'{var}_entropy'] = stats.entropy(pk)
                    else:
                        feats[f'{var}_entropy'] = 0.0
                else:
                    feats[f'{var}_entropy'] = 0.0
            except:
                feats[f'{var}_entropy'] = 0.0
            
            # ============================================================
            # ADVANCED FEATURES (7) - Optimized for short series
            # ============================================================
            
            # 1. Variance (complements std)
            feats[f'{var}_variance'] = np.var(values_clean)
            
            # 2. Coefficient of Variation (normalizes variability)
            try:
                mean_val = np.mean(values_clean)
                if abs(mean_val) > 1e-10:
                    feats[f'{var}_cv'] = np.std(values_clean) / abs(mean_val)
                else:
                    feats[f'{var}_cv'] = 0.0
            except:
                feats[f'{var}_cv'] = 0.0
            
            # 3. 90th Percentile
            feats[f'{var}_p90'] = np.percentile(values, 90)
            
            # 4. Slope (linear trend) 
            try:
                if len(values) > 1:
                    x_indices = np.arange(len(values))
                    slope, _ = np.polyfit(x_indices, values, 1)
                    feats[f'{var}_slope'] = slope
                else:
                    feats[f'{var}_slope'] = 0.0
            except:
                feats[f'{var}_slope'] = 0.0
            
            # 5. Crest Factor (max/RMS) - force/torque peaks 
            try:
                rms_val = feats[f'{var}_rms']
                max_abs = np.max(np.abs(values))
                if rms_val > 1e-10:
                    feats[f'{var}_crest_factor'] = max_abs / rms_val
                else:
                    feats[f'{var}_crest_factor'] = 0.0
            except:
                feats[f'{var}_crest_factor'] = 0.0
            
            # 6. Shape Factor (RMS/mean_abs) - signal shape
            try:
                rms_val = feats[f'{var}_rms']
                mean_abs_val = feats[f'{var}_mean_abs']
                if mean_abs_val > 1e-10:
                    feats[f'{var}_shape_factor'] = rms_val / mean_abs_val
                else:
                    feats[f'{var}_shape_factor'] = 0.0
            except:
                feats[f'{var}_shape_factor'] = 0.0
            
            # 7. Peak Frequency - dominant frequency 
            try:
                if len(values) > 2:
                    # FFT to find dominant frequency
                    fft_vals = np.fft.fft(values - np.mean(values))
                    freqs = np.fft.fftfreq(len(values))
                    
                    # Positive frequencies only
                    positive_freqs = freqs[1:len(freqs)//2 + 1]
                    fft_magnitude = np.abs(fft_vals[1:len(freqs)//2 + 1])
                    
                    if len(fft_magnitude) > 0 and np.max(fft_magnitude) > 1e-10:
                        peak_freq_idx = np.argmax(fft_magnitude)
                        peak_frequency = abs(positive_freqs[peak_freq_idx])
                        feats[f'{var}_peak_frequency'] = peak_frequency
                    else:
                        feats[f'{var}_peak_frequency'] = 0.0
                else:
                    feats[f'{var}_peak_frequency'] = 0.0
            except:
                feats[f'{var}_peak_frequency'] = 0.0
    
    except Exception as e:
        print(f"  Error in extraction for row: {e}")
        # Return default features in case of complete error
        default_features = {}
        for var in variables:
            var_features = {
                f'{var}_mean': 0.0, f'{var}_std': 0.0, f'{var}_max': 0.0, f'{var}_min': 0.0,
                f'{var}_median': 0.0, f'{var}_iqr': 0.0, f'{var}_skewness': 0.0, f'{var}_kurtosis': 0.0,
                f'{var}_rms': 0.0, f'{var}_energy': 0.0, f'{var}_power': 0.0,
                f'{var}_zero_crossing_rate': 0.0, f'{var}_first': 0.0, f'{var}_last': 0.0,
                f'{var}_peak_to_peak': 0.0, f'{var}_mean_abs': 0.0, f'{var}_entropy': 0.0,
                f'{var}_variance': 0.0, f'{var}_cv': 0.0, f'{var}_p90': 0.0, f'{var}_slope': 0.0,
                f'{var}_crest_factor': 0.0, f'{var}_shape_factor': 0.0, f'{var}_peak_frequency': 0.0
            }
            default_features.update(var_features)
        feats = default_features
        total_outliers_removed = 0
    
    # Add information about outliers removed as additional feature (optional)
    if OUTLIER_CONFIG['enable_outlier_treatment']:
        feats['total_outliers_removed'] = total_outliers_removed
    
    return pd.Series(feats)

# ====================================================================
# MAIN PROCESSING
# ====================================================================

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load dataset
print("\n LOADING ORIGINAL DATASET...")
df = pd.read_csv('./data/raw/dataset_robot.csv')
print(f" Dataset loaded: {df.shape}")
print(f"   NaN values: {df.isna().sum().sum()}")
print(f"   Classes: {df['label'].value_counts().to_dict()}")

# Initial validation
if 'label' not in df.columns:
    print(" CRITICAL ERROR: 'label' column not found!")
    exit()

# Clean problematic labels
initial_label_issues = df['label'].isna().sum()
if initial_label_issues > 0:
    print(f" WARNING: {initial_label_issues} missing labels in original dataset")
    print("   Removing rows with problematic labels...")
    df = df[df['label'].notna() & (df['label'] != '') & (df['label'] != ' ')].reset_index(drop=True)
    print(f"    Clean dataset: {df.shape}")

# Define variables
variables = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Process variable columns (convert strings to lists)
print("\n PROCESSING VARIABLE COLUMNS...")
for var in variables:
    df[var] = df[var].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(f" NaN values after transformation: {df[variables].isna().sum().sum()}")

# Prepare target
print("\n PREPARING LABELS...")
y = df['label'].copy()
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f" Classes: {le.classes_}")
print(f" Distribution: {np.bincount(y_encoded)}")

# Stratified train/test split
print(f"\n{'='*50}")
print(" STRATIFIED TRAIN/TEST SPLIT")
print(f"{'='*50}")

try:
    # Split complete DataFrame
    df_train, df_test, y_train_original, y_test_original = train_test_split(
        df, y, 
        test_size=SPLIT_CONFIG['test_size'], 
        random_state=SPLIT_CONFIG['random_state'], 
        stratify=y if SPLIT_CONFIG['stratify'] else None
    )
    
    print(f" Split completed successfully!")
    print(f"   Total: {len(df)} → Train: {len(df_train)}, Test: {len(df_test)}")
    
    print(f"\n TRAIN DISTRIBUTION:")
    train_dist = y_train_original.value_counts()
    for cls, count in train_dist.items():
        print(f"   {cls}: {count}")
    
    print(f"\n TEST DISTRIBUTION:")
    test_dist = y_test_original.value_counts()  
    for cls, count in test_dist.items():
        print(f"   {cls}: {count}")
        
except Exception as e:
    print(f" ERROR in split: {e}")
    exit()

# ====================================================================
# SAVE ORIGINAL DATASETS
# ====================================================================

print(f"\n{'='*50}")
print(" SAVING ORIGINAL DATASETS")
print(f"{'='*50}")

# Prepare original data (variables only, without labels)
X_train_original = df_train[variables].copy()
X_test_original = df_test[variables].copy()

# Save original train
train_original_path = os.path.join(OUTPUT_FOLDER, 'dataset_train_original.csv')
train_original_success = safe_save_dataset(
    X_train_original, y_train_original, 
    train_original_path, "ORIGINAL TRAIN"
)

# Save original test
test_original_path = os.path.join(OUTPUT_FOLDER, 'dataset_test_original.csv')
test_original_success = safe_save_dataset(
    X_test_original, y_test_original, 
    test_original_path, "ORIGINAL TEST"
)

if not train_original_success or not test_original_success:
    print(" CRITICAL FAILURE saving original datasets!")
    exit()

print(" Original datasets saved successfully!")

# ====================================================================
# STATISTICAL FEATURE EXTRACTION
# ====================================================================

print(f"\n{'='*50}")
print(" STATISTICAL FEATURE EXTRACTION")
print(f"{'='*50}")

print(f" Extracting 24 features per variable:")
print(f"    Basic (17): mean, std, max, min, median, iqr, skewness, kurtosis,")
print(f"                 rms, energy, power, zero_crossing_rate, first, last,")
print(f"                 peak_to_peak, mean_abs, entropy")
print(f"    Advanced (7): variance, cv, p90, slope, crest_factor, shape_factor, peak_frequency")
print(f"    Optimized for: short time series (15 points) and robotic applications")

# Add outlier feature if enabled
total_features_per_var = 24
if OUTLIER_CONFIG['enable_outlier_treatment']:
    print(f"    Additional feature: total_outliers_removed (total: {total_features_per_var * len(variables) + 1} features)")
else:
    print(f"    Total features: {total_features_per_var * len(variables)}")

# Extract features from train set
print(f"\n Extracting features from TRAIN set...")
X_train_transformed = df_train.apply(
    lambda row: extract_statistical_features(row, variables, OUTLIER_CONFIG), axis=1
)

# Extract features from test set
print(f"\n Extracting features from TEST set...")
X_test_transformed = df_test.apply(
    lambda row: extract_statistical_features(row, variables, OUTLIER_CONFIG), axis=1
)

print(f"\n Feature extraction completed!")
print(f"    Features per sample: {X_train_transformed.shape[1]}")

if OUTLIER_CONFIG['enable_outlier_treatment']:
    features_per_var = (X_train_transformed.shape[1] - 1) // len(variables)  # -1 for total_outliers_removed
    print(f"    Features per variable: {features_per_var}")
    print(f"    Outlier feature: total_outliers_removed")
    
    # Outlier removal statistics
    outliers_stats = X_train_transformed['total_outliers_removed']
    print(f"    Outliers removed - Train: min={outliers_stats.min()}, max={outliers_stats.max()}, mean={outliers_stats.mean():.2f}")
    
    outliers_test_stats = X_test_transformed['total_outliers_removed']
    print(f"    Outliers removed - Test: min={outliers_test_stats.min()}, max={outliers_test_stats.max()}, mean={outliers_test_stats.mean():.2f}")
else:
    features_per_var = X_train_transformed.shape[1] // len(variables)
    print(f"    Features per variable: {features_per_var}")

print(f"    Total variables: {len(variables)} (Fx, Fy, Fz, Tx, Ty, Tz)")

# Post-extraction validation
print(f"\n POST-EXTRACTION VALIDATION:")
validate_data_integrity(X_train_transformed, y_train_original, "TRAIN post-extraction")
validate_data_integrity(X_test_transformed, y_test_original, "TEST post-extraction")

# NaN treatment
print(f"\n NaN TREATMENT...")
X_train_transformed = X_train_transformed.replace([np.inf, -np.inf], np.nan)
X_test_transformed = X_test_transformed.replace([np.inf, -np.inf], np.nan)

# Calculate train medians for filling
train_medians = X_train_transformed.median(numeric_only=True)
X_train_clean = X_train_transformed.fillna(train_medians)
X_test_clean = X_test_transformed.fillna(train_medians)

print(f" NaNs post-cleaning - TRAIN: {X_train_clean.isna().sum().sum()}")
print(f" NaNs post-cleaning - TEST: {X_test_clean.isna().sum().sum()}")

# Final validation
print(f"\n🔍 FINAL VALIDATION:")
validate_data_integrity(X_train_clean, y_train_original, "FINAL TRAIN")
validate_data_integrity(X_test_clean, y_test_original, "FINAL TEST")

# ====================================================================
# SAVE TRANSFORMED DATASETS
# ====================================================================

print(f"\n{'='*50}")
print(" SAVING TRANSFORMED DATASETS")
print(f"{'='*50}")

# Save transformed train
train_transformed_path = os.path.join(OUTPUT_FOLDER, 'dataset_train_transformed.csv')
train_transformed_success = safe_save_dataset(
    X_train_clean, y_train_original, 
    train_transformed_path, "TRANSFORMED TRAIN"
)

# Save transformed test
test_transformed_path = os.path.join(OUTPUT_FOLDER, 'dataset_test_transformed.csv')
test_transformed_success = safe_save_dataset(
    X_test_clean, y_test_original, 
    test_transformed_path, "TRANSFORMED TEST"
)

if not train_transformed_success or not test_transformed_success:
    print(" CRITICAL FAILURE saving transformed datasets!")
    exit()

print(" Transformed datasets saved successfully!")

# ====================================================================
# FINAL REPORT
# ====================================================================

print(f"\n{'='*60}")
print(" PREPROCESSING PIPELINE COMPLETED")
print(f"{'='*60}")

print(f"\n FILES GENERATED in '{OUTPUT_FOLDER}/':")
print(f"    dataset_train_original.csv - Original train data ({len(y_train_original)} samples)")
print(f"    dataset_test_original.csv - Original test data ({len(y_test_original)} samples)")
print(f"    dataset_train_transformed.csv - Statistical features train ({X_train_clean.shape[1]} features)")
print(f"    dataset_test_transformed.csv - Statistical features test ({X_train_clean.shape[1]} features)")

print(f"\n TRANSFORMATION SUMMARY:")
print(f"    Input: {len(variables)} temporal variables (lists of ~15 points)")
print(f"    Output: {X_train_clean.shape[1]} statistical features ({features_per_var} per variable)")
print(f"    Split: {len(y_train_original)} train + {len(y_test_original)} test = {len(y_train_original) + len(y_test_original)} total")
print(f"    Classes preserved: {sorted(y_train_original.unique().tolist())}")

print(f"\n OUTLIER CONFIGURATION APPLIED:")
if OUTLIER_CONFIG['enable_outlier_treatment']:
    print(f"    Treatment: {OUTLIER_CONFIG['method'].upper()}")
    if OUTLIER_CONFIG['method'] == 'iqr':
        print(f"    IQR Multiplier: {OUTLIER_CONFIG['iqr_multiplier']}")
    elif OUTLIER_CONFIG['method'] == 'zscore':
        print(f"    Z-Score Threshold: {OUTLIER_CONFIG['zscore_threshold']}")
    elif OUTLIER_CONFIG['method'] == 'percentile':
        print(f"    Percentiles: {OUTLIER_CONFIG['percentile_lower']}% - {OUTLIER_CONFIG['percentile_upper']}%")
    print(f"    Min samples: {OUTLIER_CONFIG['min_samples_after_treatment']}")
else:
    print(f"    Outlier treatment disabled - all data preserved")

print(f"\n EXTRACTED FEATURES (24 per variable):")
print(f"    Basic statistics: mean, std, max, min, median, iqr")
print(f"    Distribution shape: skewness, kurtosis, entropy")
print(f"    Energy and magnitude: rms, energy, power, mean_abs")
print(f"    Temporal characteristics: first, last, peak_to_peak, zero_crossing_rate")
print(f"    Advanced features: variance, cv, p90, slope")
print(f"    Robotics specific: crest_factor, shape_factor")
print(f"    Frequency domain: peak_frequency")

if OUTLIER_CONFIG['enable_outlier_treatment']:
    print(f"    Quality control: total_outliers_removed")

print(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
print(f" Datasets ready for feature selection and modeling!")

# Display some statistics of generated features
if not X_train_clean.empty:
    print(f"\n GENERATED FEATURES STATISTICS:")
    print(f"    Value range: {X_train_clean.min().min():.4f} to {X_train_clean.max().max():.4f}")
    print(f"    General mean: {X_train_clean.mean().mean():.4f}")
    print(f"    Features with zero variation: {(X_train_clean.std() == 0).sum()}")
    
    # Show some example features
    print(f"\n EXAMPLE GENERATED FEATURES (first 10):")
    sample_features = X_train_clean.columns[:10].tolist()
    for feat in sample_features:
        print(f"   {feat}: {X_train_clean[feat].iloc[0]:.4f}")

print(f"\n All files saved in: {os.path.abspath(OUTPUT_FOLDER)}/")

# === SAVE CONFIGURATION USED ===
config_summary = {
    'split_config': SPLIT_CONFIG,
    'outlier_config': OUTLIER_CONFIG,
    'variables': variables,
    'output_folder': OUTPUT_FOLDER,
    'features_per_variable': features_per_var,
    'total_features': X_train_clean.shape[1],
    'total_samples': {
        'train': len(y_train_original),
        'test': len(y_test_original)
    },
    'classes': sorted(y_train_original.unique().tolist())
}

# Save configuration to file
import json
config_path = os.path.join(OUTPUT_FOLDER, 'preprocessing_config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\n Configuration saved in: preprocessing_config.json")
print(f" To replicate: use the same SPLIT_CONFIG and OUTLIER_CONFIG settings")
