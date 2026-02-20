import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ==========================================
# 1. PHYSICS MODELS (The Equations)
# ==========================================

def fireball_model(t, A, t0, B):
    """
    Early Rise Model: Flux ~ (t - t0)^2
    Used for 'rise_fireball_error'
    """
    t_safe = np.array(t)
    return A * (t_safe - t0)**2 + B

def power_law_model(t, A, t0, B):
    """
    Late Decay Model: Flux ~ (t - t0)^-5/3
    Used for 'tde_power_law_error'
    """
    t_safe = np.array(t)
    with np.errstate(invalid='ignore'):
        decay = np.power(t_safe - t0, -5/3)
    decay = np.nan_to_num(decay, nan=0.0)
    return A * decay + B

# ==========================================
# 2. ANATOMY PLOTS (Visualizing the Math)
# ==========================================

def plot_fireball_anatomy(df, save_dir):
    print("   [+] Generating Anatomy: Fireball Rise (t^2)...")
    
    # Find a TDE with the most pre-peak data points
    tde_ids = df[df['target'] == 1]['object_id'].unique()
    best_id = None
    best_score = -1

    for oid in tde_ids:
        subset = df[df['object_id'] == oid]
        if subset.empty: 
            continue
        
        # Use g-band (usually clearly visible in rise)
        g_data = subset[subset['filter'] == 'g']
        if g_data.empty: 
            continue
        
        peak_idx = g_data['flux'].idxmax()
        t_peak = g_data.loc[peak_idx, 'mjd']
        
        # Count points before peak
        pre_peak = g_data[g_data['mjd'] < t_peak]
        if len(pre_peak) > best_score:
            best_score = len(pre_peak)
            best_id = oid

    if not best_id:
        print("[!] No good fireball candidate found.")
        return

    # Plotting
    data = df[df['object_id'] == best_id]
    plot_data = data[data['filter'] == 'g'].sort_values('mjd')
    
    peak_idx = plot_data['flux'].idxmax()
    t_peak = plot_data.loc[peak_idx, 'mjd']
    y_peak = plot_data.loc[peak_idx, 'flux']
    
    # Rise Phase only
    rise_data = plot_data[plot_data['mjd'] <= t_peak]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Full Curve Ghost
    plt.errorbar(plot_data['mjd'], plot_data['flux'], yerr=plot_data['flux_err'], 
                 fmt='o', color='lightgrey', label='Full Lightcurve')
    
    # Highlight Rise
    plt.errorbar(rise_data['mjd'], rise_data['flux'], yerr=rise_data['flux_err'], 
                 fmt='o', color='#d35400', label='Rise Phase Data')

    # Fit t^2
    try:
        p0 = [0.01, rise_data['mjd'].min() - 5, 0]
        popt, _ = curve_fit(fireball_model, rise_data['mjd'], rise_data['flux'], p0=p0, maxfev=5000)
        
        t_line = np.linspace(rise_data['mjd'].min() - 5, t_peak, 100)
        y_model = fireball_model(t_line, *popt)
        
        plt.plot(t_line, y_model, color='#e67e22', lw=3, label=r'Fireball Model ($t^2$)')
        
        # Visualize Error (Residuals)
        # shade area between points and line? Or just text.
        plt.text(t_peak - 10, y_peak * 0.8, 
                 "rise_fireball_error:\n(Residuals from this curve)", 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
                 
    except Exception as e:
        print(f"Fireball fit failed: {e}")

    plt.title(f"Feature: Fireball Rise (Object {best_id})", fontsize=14, fontweight='bold')
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (g-band)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "anatomy_fireball.png"), dpi=200)
    plt.close()


def plot_decay_anatomy(df, save_dir):
    print("   [+] Generating Anatomy: Power Law Decay [Long-Tail Mode]...")
    
    # 1. FIND THE TDE WITH THE LONGEST TAIL
    # A short tail looks linear. A long tail shows the -5/3 curve.
    tde_ids = df[df['target'] == 1]['object_id'].unique()
    
    best_id = None
    max_duration = 0
    
    for oid in tde_ids:
        subset = df[df['object_id'] == oid]
        g_data = subset[subset['filter'] == 'g']
        if g_data.empty:
             continue
        
        peak_idx = g_data['flux'].idxmax()
        t_peak = g_data.loc[peak_idx, 'mjd']
        t_end = g_data['mjd'].max()
        
        tail_duration = t_end - t_peak
        
        # We want a long tail, but also enough points to be real
        if tail_duration > max_duration and len(g_data) > 20:
            max_duration = tail_duration
            best_id = oid

    if not best_id:
         return

    # 2. PREPARE DATA
    data = df[df['object_id'] == best_id]
    plot_data = data[data['filter'] == 'g'].sort_values('mjd')
    
    peak_idx = plot_data['flux'].idxmax()
    t_peak = plot_data.loc[peak_idx, 'mjd']
    y_peak = plot_data.loc[peak_idx, 'flux']
    
    decay_data = plot_data[plot_data['mjd'] >= t_peak]
    
    # 3. FIT THE MODEL
    # We fit A * (t - t0)^-5/3 + B
    try:
        p0 = [y_peak, t_peak - 10, 0]
        # Constrain t0 to be before the peak
        bounds = ([0, -np.inf, -np.inf], [np.inf, t_peak, np.inf])
        popt, _ = curve_fit(power_law_model, decay_data['mjd'], decay_data['flux'], 
                          p0=p0, bounds=bounds, maxfev=10000)
        fit_success = True
    except Exception:
        fit_success = False

    # 4. PLOT
    plt.figure(figsize=(10, 6))
    
    # Raw Data
    plt.errorbar(plot_data['mjd'], plot_data['flux'], yerr=plot_data['flux_err'], 
                 fmt='o', color='lightgrey', alpha=0.5, label='Pre-Peak / Noise')
    plt.errorbar(decay_data['mjd'], decay_data['flux'], yerr=decay_data['flux_err'], 
                 fmt='o', color='#2980b9', alpha=0.8, label='Decay Phase Data')
                 
    if fit_success:
        # Generate smooth curve extending slightly past data
        t_model = np.linspace(t_peak, plot_data['mjd'].max() + 50, 200)
        y_model = power_law_model(t_model, *popt)
        
        # Plot Best Fit Line
        plt.plot(t_model, y_model, color='#3498db', lw=4, label=r'Best Fit Model')
        
        # Plot "Pure" Theoretical Curve for comparison (dashed)
        # This shows what a "Perfect" -5/3 decay looks like from that peak
        # We normalize it to match the peak height
        y_theory = y_peak * ((t_model - t_peak + 10) / 10)**(-5/3)
        # Shift theory curve to align visually with data baseline if needed
        y_theory += popt[2] 
        
        plt.plot(t_model, y_theory, color='#e74c3c', lw=2, linestyle='--', alpha=0.7, 
                 label=r'Theoretical $t^{-5/3}$ Limit')

        # Annotation Box
        plt.text(t_peak + 50, y_peak * 0.7, 
                 "tde_power_law_error:\n(Measures deviation from the blue line)", 
                 fontsize=11, bbox=dict(facecolor='white', edgecolor='#3498db', alpha=0.9))

    plt.title(f"Feature Anatomy: Power Law Decay (Object {best_id})", fontsize=14, fontweight='bold')
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (g-band)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.savefig(os.path.join(save_dir, "anatomy_decay.png"), dpi=300)
    plt.close()
    print(f"   [+] Saved anatomy_decay.png (Object {best_id})")

def plot_gp_anatomy(df, save_dir):
    print("   [+] Generating Anatomy: Gaussian Process (Length Scale)...")
    
    # Use the "Textbook TDE" if available, otherwise find a fallback
    target_id = 'talraph_gwael_lunt'
    if target_id not in df['object_id'].values:
        target_id = df[df['target'] == 1]['object_id'].unique()[0]

    # Get Data (g-band only for clarity)
    data = df[(df['object_id'] == target_id) & (df['filter'] == 'g')].sort_values('mjd')
    if data.empty: 
        return

    # Prepare Data for GP (Normalization is key for stability)
    X = data['mjd'].values.reshape(-1, 1)
    y = data['flux'].values
    y_err = data['flux_err'].values
    
    # Simple Normalization
    X_mean = X.mean()
    y_mean = y.mean()
    y_std = y.std() + 1e-6
    
    X_norm = X - X_mean
    y_norm = (y - y_mean) / y_std
    
    # Define Kernel: RBF (Smooth Signal) + WhiteKernel (Noise Estimation)
    # We initialize length_scale to 20 days (typical for transients)
    kernel = 1.0 * RBF(length_scale=20.0, length_scale_bounds=(1e-1, 1e3)) + \
             WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1))
             
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(y_err/y_std)**2, normalize_y=False)
    gp.fit(X_norm, y_norm)

    # Predict on a dense grid for the smooth "cloud"
    X_plot = np.linspace(X_norm.min()-20, X_norm.max()+20, 200).reshape(-1, 1)
    y_pred_norm, sigma_norm = gp.predict(X_plot, return_std=True)
    
    # De-normalize for plotting
    X_plot_real = X_plot + X_mean
    y_pred_real = (y_pred_norm * y_std) + y_mean
    sigma_real = sigma_norm * y_std

    # Extract Learned Length Scale
    learned_ls = gp.kernel_.get_params()['k1__k2__length_scale']
    
    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    
    # 1. Observed Data
    plt.errorbar(data['mjd'], data['flux'], yerr=data['flux_err'], fmt='o', color='black', 
                 label='Observed Flux', zorder=3)
    
    # 2. GP Mean Prediction (The "Smooth" fit)
    plt.plot(X_plot_real, y_pred_real, color='#2980b9', lw=2, label='GP Mean Prediction', zorder=2)
    
    # 3. Uncertainty Cloud (The "Confidence")
    plt.fill_between(X_plot_real.ravel(), y_pred_real - sigma_real, y_pred_real + sigma_real, 
                     color='#2980b9', alpha=0.2, label='GP Uncertainty (Sigma)', zorder=1)

    # Annotation
    plt.text(X_mean, y_mean + y_std, 
             f"Learned Length Scale: {learned_ls:.1f} days\n(Feature 'ls_time' = Explosion Duration)", 
             fontsize=11, bbox=dict(facecolor='white', edgecolor='#2980b9', alpha=0.9), ha='center')

    plt.title(f"Feature Anatomy: Gaussian Process Fit | Object {target_id} | Type: TDE", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (g-band)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    plt.savefig(os.path.join(save_dir, "anatomy_gp.png"), dpi=200)
    plt.close()
    print(f"   [+] Saved anatomy_gp.png (Object {target_id})")


# ==========================================
# 3. DISTRIBUTION PLOTS (The "All Features" Loop)
# ==========================================

def plot_all_distributions(df, save_dir):
    print("   [+] Generating Distributions for ALL features...")
    
    # YOUR FULL LIST
    features_to_plot = [
        # GP
        'amplitude', 'ls_time', 'ls_wave',
        # Morphology
        'rise_time', 'fade_time', 'fwhm', 'rise_fade_ratio', 'compactness', 'rise_slope',
        # Physics
        'tde_power_law_error', 'template_chisq_tde', 'linear_decay_slope',
        # Color
        'mean_color_gr', 'std_color_gr', 'mean_color_gr_pre', 'color_slope_gr_pre', 'blue_energy_fraction',
        # Stats
        'total_radiated_energy_proxy', 'negative_flux_fraction',
        # Shape Gates
        'percentile_ratio_20_50', 'percentile_ratio_80_max',
        # Advanced
        'rise_fireball_error', 'pre_peak_var', 'reduced_chi_square', 'fade_shape_correlation',
        'baseline_ratio', 'color_cooling_rate', 'color_slope_gr',
        # Snapshots
        'ug_peak', 'gr_peak', 'ur_peak', 'flux_ratio_ug', 'flux_ratio_gr',
        # Robust
        'flux_kurtosis', 'flux_skew', 'robust_duration', 'duty_cycle'
    ]

    for feat in features_to_plot:
        if feat not in df.columns:
            continue
            
        # Clean Data (Remove Inf/NaN)
        clean_df = df[[feat, 'Class']].replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_df.empty: 
            continue
        
        plt.figure(figsize=(8, 4))
        sns.set_style("ticks")
        
        # Robust Scaling: Clip top/bottom 1% to ignore outliers
        q_low = clean_df[feat].quantile(0.01)
        q_high = clean_df[feat].quantile(0.99)
        plot_data = clean_df[(clean_df[feat] >= q_low) & (clean_df[feat] <= q_high)]
        
        if plot_data.empty: 
            continue  # Skip if all data was outliers (unlikely)

        # Plot TDE vs Other
        sns.kdeplot(data=plot_data[plot_data['Class']=='TDE'], x=feat, 
                    fill=True, color='#e74c3c', alpha=0.5, label='TDE')
        sns.kdeplot(data=plot_data[plot_data['Class']=='Other'], x=feat, 
                    fill=False, color='#34495e', linewidth=2, label='Noise/Other')
        
        plt.title(f"Separation: {feat}", fontweight='bold')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, f"dist_{feat}.png"), dpi=150)
        plt.close()

def plot_color_evolution(df, save_dir):
    print("   [+] Generating Anatomy: Multiband Color Evolution...")
    
    # Find a TDE with data in ALL bands (or at least u, g, r)
    tde_ids = df[df['target'] == 1]['object_id'].unique()
    best_id = None
    max_points = 0
    
    for oid in tde_ids:
        subset = df[df['object_id'] == oid]
        # Check for u-band (UV) presence, as it's key for TDEs
        if 'u' in subset['filter'].values and len(subset) > max_points:
            max_points = len(subset)
            best_id = oid

    if not best_id: 
        return

    data = df[df['object_id'] == best_id]
    
    # Center time on the overall peak
    peak_idx = data['flux'].idxmax()
    t_peak = data.loc[peak_idx, 'mjd']
    
    plt.figure(figsize=(10, 6))
    
    # Standard LSST Colors
    colors = {'u': '#8e44ad', 'g': '#2ecc71', 'r': '#e74c3c', 
              'i': '#f39c12', 'z': '#34495e', 'y': '#95a5a6'}
    
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        band_data = data[data['filter'] == band].sort_values('mjd')
        if band_data.empty: 
            continue
        
        plt.errorbar(band_data['mjd'] - t_peak, band_data['flux'], yerr=band_data['flux_err'], 
                     fmt='o', color=colors[band], label=f'{band}-band', alpha=0.7)
        
        # Connect lines for visibility
        plt.plot(band_data['mjd'] - t_peak, band_data['flux'], color=colors[band], alpha=0.3)

    plt.title(f"Multiband Lightcurve (Color Evolution) | {best_id} | Type: TDE", fontsize=14, fontweight='bold')
    plt.xlabel("Days from Peak")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.xlim(-50, 150)
    
    plt.savefig(os.path.join(save_dir, "anatomy_colors.png"), dpi=200)
    plt.close()

# ==========================================
# 4. WRAPPER FOR MASTER SCRIPT
# ==========================================

def generate_feature_plots(df_curves, df_feats, anatomy_dir, dist_dir):
    print("--- STARTING COMPREHENSIVE FEATURE PLOTTING ---")
    
    # 1. Run Anatomy Plots (Visual Explanations using Raw Curves)
    plot_fireball_anatomy(df_curves, anatomy_dir)
    plot_decay_anatomy(df_curves, anatomy_dir)
    plot_gp_anatomy(df_curves, anatomy_dir)
    plot_color_evolution(df_curves, anatomy_dir)
    
    # 2. Run Distributions (The Full List using Extracted Features)
    plot_all_distributions(df_feats, dist_dir)
    
    print("--- DONE ---")
    print(f"Check '{anatomy_dir}' for visual explanations.")
    print(f"Check '{dist_dir}' for the 30+ distribution plots.")