'''
realtime/T2TDEClassifierEnsemble.py
Author: maia.advance, maymeridian
Description: TDEClassifierEnsemble implements AbsTiedStateT2Unit from Ampel-HU-Astro as base class, for use with real data. 
'''
from ampel.contrib.hu.t2.T2BaseClassifier import T2BaseClassifier
from ampel.abstract.AbsTiedStateT2Unit import AbsTiedStateT2Unit
from ampel.view.LightCurve import LightCurve

from ampel.content.DataPoint import DataPoint
from ampel.content.T1Document import T1Document
from ampel.view.T2DocView import T2DocView
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson

# ---- Model Dependencies ----
from extinction import fitzpatrick99
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import kurtosis, skew, linregress
from joblib import Parallel, delayed

import numpy as np
import pandas as pd


class T2TDEClassifierEnsemble(AbsTiedStateT2Unit): 

    # need to configure filepath via yaml file. 
    model_filepath: str

    # Effective wavelengths (Angstroms) for LSST filters
    FILTER_WAVELENGTHS = {
        'u': 3641,
        'g': 4704,
        'r': 6155,
        'i': 7504,
        'z': 8695,
        'y': 10056
    }

        
    # Subset 1: Morphology & Temporal Evolution Features
    MORPHOLOGY_FEATURES = [
        'rest_rise_time',
        'rest_fade_time',
        'rest_fwhm',
        'ls_time',
        'rise_fade_ratio',
        'compactness',
        'rise_slope',
        'flux_kurtosis',
        'robust_duration',
        'duty_cycle',
        'pre_peak_var',
        'amplitude'
    ]

    # Subset 2: Physics & Color Metrics
    PHYSICS_FEATURES = [
        'tde_power_law_error',
        'template_chisq_tde',
        'linear_decay_slope',
        'mean_color_gr',
        'std_color_gr',
        'mean_color_gr_pre',
        'color_slope_gr_pre',
        'blue_energy_fraction',
        'total_radiated_energy',
        'color_monotonicity',
        'negative_flux_fraction',
        'rise_fireball_error',
        'reduced_chi_square',
        'ls_wave',
        'fade_shape_correlation',
        'baseline_ratio',
        'color_cooling_rate',
        'color_slope_gr',
        'flux_ratio_ug',
        'flux_ratio_gr',
        'ug_peak',
        'gr_peak',
        'ur_peak',
        'redshift',
        'absolute_magnitude_proxy',
        'log_tde_error'
    ]


    def post_init(self) -> None:
        super().post_init()
        self.model, self.threshold = self.load_model(self.model_filepath)
        self.logger.info(f"Loaded TDE model from {self.model_filepath}")


    def load_model(self): 
        if not os.path.exists(MODEL_PATH):
            self.logger.info(f"Failed to Load TDE Ensemble model from {self.model_filepath}")
            return

        model = joblib.load(MODEL_PATH)
        
        # temporarily hard-coding the model threshold, since we will not be changing 
        # it until retraining; which would require other changes anyway.
        threshold = 0.3999999999999999 

    # -------------------
    #   PROCESSING IMPL.
    # -------------------
   
    def apply_deextinction(self, df, log_df):
        """
        Applies Galactic Extinction correction (Milky Way dust) to fluxes.
        Uses the Fitzpatrick dust law.
        """
        if 'Flux_Corrected' in df.columns:
            return df
        if 'EBV' not in df.columns:
            if 'EBV' in log_df.columns:
                ebv_map = log_df.set_index('object_id')['EBV']
                df['EBV'] = df['object_id'].map(ebv_map)
            else:
                # Fallback if no EBV provided
                df['Flux_Corrected'] = df['Flux']
                df['Flux_err_Corrected'] = df['Flux_err']
                return df

        unique_filters = list(FILTER_WAVELENGTHS.keys())
        unique_wls = np.array([FILTER_WAVELENGTHS[f] for f in unique_filters], dtype=float)
        ext_factors = fitzpatrick99(unique_wls, 1.0)
        ext_map = dict(zip(unique_filters, ext_factors))

        # A_lambda = R_v * E(B-V) * k(lambda). Assuming R_v = 3.1
        a_lambda = df['Filter'].map(ext_map) * (df['EBV'] * 3.1)
        correction_factor = 10**(a_lambda / 2.5)

        df['Flux_Corrected'] = df['Flux'] * correction_factor
        df['Flux_err_Corrected'] = df['Flux_err'] * correction_factor
        return df


    def apply_quality_cuts(self, lc_df):
        """
        Pre-processing filter to remove garbage data before GP fitting.
        - Calculates SNR.
        - Drops objects with fewer than 2 positive flux detections.
        """
        if 'SNR' not in lc_df.columns:
            safe_err = lc_df['Flux_err'].replace(0, 1e-5)
            lc_df['SNR'] = lc_df['Flux'] / safe_err

        valid_mask = (lc_df['Flux'] > 0)
        counts = lc_df[valid_mask].groupby('object_id').size()
        keep_ids = counts[counts >= 2].index

        return lc_df[lc_df['object_id'].isin(keep_ids)].copy()


    # -----------------------
    # Functions used
    # within get_gp_features: 
    # -----------------------

    def fit_2d_gp(self, obj_df):
        """
        Fits a 2D Gaussian Process (Time + Wavelength) to the light curve.
        This allows us to interpolate flux at any time for any filter, handling
        the sparse/irregular cadence of LSST data.
        """
        if 'Flux_Corrected' in obj_df.columns:
            y = obj_df['Flux_Corrected'].values
            y_err = obj_df['Flux_err_Corrected'].values
        else:
            y = obj_df['Flux'].values
            y_err = obj_df['Flux_err'].values

        # Input: Time (MJD) and Wavelength (Angstroms)
        X = np.column_stack([obj_df['Time (MJD)'].values, obj_df['Filter'].map(FILTER_WAVELENGTHS).values])

        # Normalize Y to stabilize the optimizer
        y_scale = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
        y_norm = y / y_scale
        y_err_norm = y_err / y_scale

        # Kernel: Constant * Matern(nu=1.5).
        # Matern 1.5 allows for rougher functions (explosions) compared to RBF.
        kernel = ConstantKernel(1.0) * Matern(length_scale=[100, 6000], nu=1.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_norm**2, n_restarts_optimizer=0, random_state=42)
        gp.fit(X, y_norm)

        return gp, y_scale
    

    def calculate_physics_wars(self, t_grid, y_pred_g, peak_idx, peak_time, peak_flux):
        """
        Tests the light curve against known physical models:
        1. Power Law Decay (t^-5/3): The signature of tidal disruption.
        2. Fireball Rise (t^2): The signature of expanding debris.
        3. Log-Linear Slope: A robust fallback for decay measurement.
        """
        # 1. TDE FIT (Power Law)
        post_peak_indices = np.where(t_grid > peak_time)[0]
        tde_error = 10.0
        linear_decay_slope = 0.0

        if len(post_peak_indices) > 5 and peak_flux > 0:
            y_real_fade = y_pred_g[post_peak_indices]
            t_fade = t_grid[post_peak_indices]
            dt = (t_fade - peak_time) + 10  # Add offset to avoid log(0)

            # Ideal TDE Model
            y_ideal_tde = peak_flux * (dt / dt[0])**(-1.67)
            residuals_tde = (y_real_fade - y_ideal_tde) / peak_flux
            tde_error = np.mean(residuals_tde**2)

            # Robust Linear Slope (Log-Log Space)
            try:
                log_t = np.log(dt)
                log_y = np.log(y_real_fade + 1e-9)
                slope, _, _, _, _ = linregress(log_t, log_y)
                linear_decay_slope = slope
            except Exception:
                linear_decay_slope = 0.0

        # 2. RISE PHYSICS (Fireball Model)
        pre_peak_indices = np.where(t_grid < peak_time)[0]
        rise_fireball_error = 10.0
        pre_peak_var = 0.0

        if len(pre_peak_indices) > 5 and peak_flux > 0:
            y_real_rise = y_pred_g[pre_peak_indices]
            t_rise = t_grid[pre_peak_indices]

            if len(t_rise) > 3:
                try:
                    # Fit 2nd order polynomial (parabola)
                    coeffs = np.polyfit(t_rise, y_real_rise, 2)
                    p = np.poly1d(coeffs)
                    residuals_rise = (y_real_rise - p(t_rise)) / peak_flux
                    rise_fireball_error = np.mean(residuals_rise**2)
                    pre_peak_var = np.var(residuals_rise)  # Detects "bumps" before peak
                except Exception:
                    pass

        # 3. Shape Analysis
        fade_correlation = 0.0
        if len(post_peak_indices) > 2:
            fade_correlation = np.corrcoef(t_grid[post_peak_indices], y_pred_g[post_peak_indices])[0, 1]

        # Full Width Half Max (FWHM) calculation
        half_max = peak_flux / 2.0
        rise_idx_candidates = np.where((y_pred_g[:peak_idx] <= half_max))[0]
        t_half_rise = t_grid[rise_idx_candidates[-1]] if len(rise_idx_candidates) > 0 else t_grid[0]

        fade_idx_candidates = np.where((y_pred_g[peak_idx:] <= half_max))[0]
        t_half_fade = t_grid[peak_idx + fade_idx_candidates[0]] if len(fade_idx_candidates) > 0 else t_grid[-1]

        fwhm = t_half_fade - t_half_rise

        return tde_error, linear_decay_slope, rise_fireball_error, fade_correlation, fwhm, pre_peak_var

    
    def calculate_template_matching(self, t_grid, y_pred_g, peak_idx, peak_time, peak_flux):
        """
        Calculates the 'Template Chi-Squared' error.
        Fits a normalized "Ideal TDE" shape (fast rise, power-law decay) to the data.
        Crucial for identifying faint TDEs where direct curve fitting fails.
        """
        post_peak_indices = np.where(t_grid > peak_time)[0]
        match_tde = 10.0  # Default high error

        if len(post_peak_indices) > 5 and peak_flux > 0:
            y_fade = y_pred_g[post_peak_indices]
            t_fade = t_grid[post_peak_indices] - peak_time

            # Normalize Flux and Time
            y_norm = y_fade / peak_flux
            half_idx = np.argmax(y_norm < 0.5)

            if half_idx > 0:
                t_half = t_fade[half_idx]
                if t_half > 0.1:
                    t_norm = t_fade / t_half  # t = 1.0 is the half-life

                    # TDE Template approximation: 1 / (1 + x)^1.67
                    y_temp_tde = 1.0 / (1.0 + (t_norm * (2**(1/1.67) - 1)))**1.67

                    mask = t_norm < 3.0  # Only fit the core shape

                    if mask.sum() > 2:
                        match_tde = np.sqrt(np.mean((y_norm[mask] - y_temp_tde[mask])**2))

        return match_tde

    # ------------------------
    #  End of get_gp_features
    # ------------------------


    def get_gp_features(self, obj_id, obj_df):
        """
        Extracts all features for a single object.
        Combines GP interpolation, physics metrics, morphology, and color evolution.
        """
        try:
            gp, y_scale = self.fit_2d_gp(obj_df)
        except Exception as e:
            #print(f"GP Extraction failed for {obj_id}: {e}") # Removing these objects improves model
            return None

        # Kernel Params
        # ls_time: Timescale of variability (TDEs are slower than some SNe, faster than AGNs)
        # ls_wave: Coherence across bands (TDEs are highly coherent)
        params = gp.kernel_.get_params()
        try:
            ls_time = params.get('k2__length_scale', [0, 0])[0]
            ls_wave = params.get('k2__length_scale', [0, 0])[1]
            amplitude = np.sqrt(params.get('k1__constant_value', 0)) * y_scale
        except Exception:
            ls_time, ls_wave, amplitude = 0, 0, 0

        if 'Flux_Corrected' in obj_df.columns:
            flux_data = obj_df['Flux_Corrected']
            flux_err = obj_df['Flux_err_Corrected']
        else:
            flux_data = obj_df['Flux']
            flux_err = obj_df['Flux_err']


        # Measures the time span of significant activity.
        # Long duration + High Duty Cycle = Likely AGN.
        significant_negative = (flux_data < -3 * flux_err)
        negative_flux_fraction = significant_negative.sum() / len(flux_data) if len(flux_data) > 0 else 0.0

        significant_mask = flux_data > (3 * flux_err)
        detection_times = obj_df.loc[significant_mask, 'Time (MJD)']
        total_survey_span = obj_df['Time (MJD)'].max() - obj_df['Time (MJD)'].min()

        if len(detection_times) > 4:
            t_10 = np.percentile(detection_times, 10)
            t_90 = np.percentile(detection_times, 90)
            robust_duration     = t_90 - t_10
            duty_cycle = robust_duration / total_survey_span if total_survey_span > 0 else 0
        else:
            robust_duration = 0.0
            duty_cycle = 0.0


        flux_kurtosis = kurtosis(flux_data, fisher=True)  # Spikiness
        flux_skew = skew(flux_data)  # Asymmetry

        # GP Predictions
        # Predict light curve on a dense grid for smooth feature extraction
        t_min, t_max = obj_df['Time (MJD)'].min(), obj_df['Time (MJD)'].max()
        t_grid = np.linspace(t_min, t_max, 100)
        g_wave = FILTER_WAVELENGTHS['g']

        # Predict in 'g' band (usually the most sensitive)
        X_pred_g = np.column_stack([t_grid, np.full_like(t_grid, g_wave)])
        y_pred_g = gp.predict(X_pred_g) * y_scale

        # Calculate Chi-Square of GP Fit
        X_obs = np.column_stack([obj_df['Time (MJD)'].values, obj_df['Filter'].map(FILTER_WAVELENGTHS).values])
        y_gp_pred = gp.predict(X_obs) * y_scale
        safe_err = np.where(flux_err <= 0, 1e-5, flux_err)
        chi_sq_terms = ((flux_data - y_gp_pred) / safe_err) ** 2
        reduced_chi_square = np.mean(chi_sq_terms)
        reduced_chi_square = min(reduced_chi_square, 1000.0)

        # Peak Finding
        peak_idx = np.argmax(y_pred_g)
        peak_time = t_grid[peak_idx]
        peak_flux = y_pred_g[peak_idx]
        threshold = peak_flux / 2.512

        # Shape Metrics
        # Fixes issues with "Plateau" light curves in Fold 3.
        positive_flux = y_pred_g[y_pred_g > 0]
        if len(positive_flux) > 0:
            perc_20 = np.percentile(positive_flux, 20)
            perc_50 = np.percentile(positive_flux, 50)
            perc_80 = np.percentile(positive_flux, 80)
            percentile_ratio_20_50 = perc_20 / (perc_50 + 1e-9)
            percentile_ratio_80_max = perc_80 / (peak_flux + 1e-9)
        else:
            percentile_ratio_20_50 = 0.0
            percentile_ratio_80_max = 0.0

        # Rise/Fade Times
        pre_peak = y_pred_g[:peak_idx]
        t_pre = t_grid[:peak_idx]

        if len(pre_peak) > 0 and np.any(pre_peak < threshold):
            drop_idx = np.where(pre_peak < threshold)[0][-1]
            rise_time = peak_time - t_pre[drop_idx]
        else:
            rise_time = peak_time - t_min

        post_peak = y_pred_g[peak_idx:]
        t_post = t_grid[peak_idx:]
        if len(post_peak) > 0 and np.any(post_peak < threshold):
            drop_idx = np.where(post_peak < threshold)[0][0]
            fade_time = t_post[drop_idx] - peak_time
        else:
            fade_time = t_max - peak_time

        # Physics / Model Fitting
        tde_error, linear_decay_slope, rise_error, fade_shape, fwhm, pre_peak_var = self.calculate_physics_wars(t_grid, y_pred_g, peak_idx, peak_time, peak_flux)
        match_tde = self.calculate_template_matching(t_grid, y_pred_g, peak_idx, peak_time, peak_flux)

        # Color Evolution
        # Extracting multi-band behavior at the peak and during the fade.
        def get_val(t, band):
            val = gp.predict([[t, FILTER_WAVELENGTHS[band]]])[0] * y_scale
            return val if val > 0 else 1e-5

        val_u = get_val(peak_time, 'u')
        val_g = get_val(peak_time, 'g')
        val_r = get_val(peak_time, 'r')

        # Colors
        flux_ratio_ug = val_u / val_g
        flux_ratio_gr = val_g / val_r
        ug_peak = -2.5 * np.log10(val_u / val_g)
        gr_peak = -2.5 * np.log10(val_g / val_r)
        ur_peak = -2.5 * np.log10(val_u / val_r)

        # 1. Post-Peak Color (Fade)
        # TDEs maintain Blue colors (negative g-r) while fading. SNe redden.
        t_samples = np.linspace(peak_time, peak_time + fade_time, 5)
        g_samples = [get_val(t, 'g') for t in t_samples]
        r_samples = [get_val(t, 'r') for t in t_samples]
        gr_colors = [-2.5 * np.log10(g/r) for g, r in zip(g_samples, r_samples)]

        mean_color_gr = np.mean(gr_colors)
        std_color_gr = np.std(gr_colors)

        try:
            slope, _, _, _, _ = linregress(np.arange(5), gr_colors)
            color_slope_gr = slope
        except Exception:
            color_slope_gr = 0.0

        # 2. Pre-Peak Color (Rise)
        # TDEs form a disk and heat up/stay hot during rise.
        t_samples_pre = np.linspace(peak_time - rise_time, peak_time, 5)
        g_samples_pre = [get_val(t, 'g') for t in t_samples_pre]
        r_samples_pre = [get_val(t, 'r') for t in t_samples_pre]
        gr_colors_pre = [-2.5 * np.log10(g/r) for g, r in zip(g_samples_pre, r_samples_pre)]

        mean_color_gr_pre = np.mean(gr_colors_pre)

        try:
            slope_pre, _, _, _, _ = linregress(np.arange(5), gr_colors_pre)
            color_slope_gr_pre = slope_pre
        except Exception:
            color_slope_gr_pre = 0.0

        # Cooling Rate
        t_fade_pt = peak_time + (fade_time/2)
        gr_fade = -2.5 * np.log10(get_val(t_fade_pt, 'g') / get_val(t_fade_pt, 'r'))
        color_cooling_rate = gr_fade - gr_peak

        # 3. Blue Energy Fraction
        # Calculates total integrated energy in U+G bands vs total energy.
        def get_area(band):
            y_band = gp.predict(np.column_stack([t_grid, np.full_like(t_grid, FILTER_WAVELENGTHS[band])])) * y_scale
            return np.trapezoid(y_band, t_grid)

        area_u = get_area('u')
        area_g = get_area('g')
        total_area = area_u + area_g + get_area('r') + get_area('i')

        blue_energy_fraction = (area_u + area_g) / (total_area + 1e-9)

        # Derived Morph.
        rise_fade_ratio = rise_time / fade_time if fade_time > 0 else 0
        area_under_curve = np.trapezoid(y_pred_g, t_grid)
        compactness = area_under_curve / peak_flux if peak_flux > 0 else 0
        rise_slope = amplitude / rise_time if rise_time > 1 else amplitude

        baseline_window = int(len(y_pred_g) * 0.15)
        baseline_flux = np.median(y_pred_g[:baseline_window])
        baseline_ratio = baseline_flux / peak_flux if peak_flux > 0 else 0

        return {
            'object_id': obj_id,

            # Gaussian Process Kernel Metrics
            'amplitude': amplitude,          # Peak flux height. TDEs are often intrinsically very bright.
            'ls_time': ls_time,              # GP Length Scale (Time). Measures evolutionary speed.
            'ls_wave': ls_wave,              # GP Length Scale (Wavelength). Measures SED coherence.

            # Temporal Morph.
            'rise_time': rise_time,          # Detection to Peak. TDEs rise sharply.
            'fade_time': fade_time,          # Peak to threshold. TDEs fade slowly (t^-5/3).
            'fwhm': fwhm,                    # Full Width Half Max. "Spikiness" metric.
            'rise_fade_ratio': rise_fade_ratio,  # Asymmetry Check. TDEs < 1 (Rise < Fade).
            'compactness': compactness,      # Area/Peak. Distinguishes "Blocky" AGNs from "Peaked" Transients.
            'rise_slope': rise_slope,        # Explosion violence: Amplitude / Rise Time.

            # Physics / Model Fitting
            'tde_power_law_error': tde_error,        # The "Smoking Gun": Residuals from t^-5/3 gravity decay.
            'template_chisq_tde': match_tde,         # Shape Matching: Error against normalized TDE template.
            'linear_decay_slope': linear_decay_slope,  # Robust Backup: Slope in log-log space.

            # Color Evolution
            # TDEs maintain Blue/Hot colors. SNe cool/Redden.
            'mean_color_gr': mean_color_gr,          # Avg Post-Peak Color. TDEs stay Blue (negative).
            'std_color_gr': std_color_gr,            # Color Stability. TDEs are stable, SNe vary.
            'mean_color_gr_pre': mean_color_gr_pre,   # Avg Pre-Peak Color. TDEs form hot disks early.
            'color_slope_gr_pre': color_slope_gr_pre,  # Pre-Peak Color Change.
            'blue_energy_fraction': blue_energy_fraction,  # UV/Blue Dominance. (u+g) / Total Energy.

            # Statistical Metrics
            'total_radiated_energy_proxy': total_area,  # Total Integrated Flux.
            'color_monotonicity': 0.0,
            'negative_flux_fraction': negative_flux_fraction,  # Noise/Garbage detection.

            # Shape Features
            'percentile_ratio_20_50': percentile_ratio_20_50,  # "Fatness" of the base.
            'percentile_ratio_80_max': percentile_ratio_80_max,  # "Flatness" of the peak.

            'rise_fireball_error': rise_error,       # Fireball Test (t^2).
            'pre_peak_var': pre_peak_var,            # Smoothness before peak.
            'reduced_chi_square': reduced_chi_square,# Smoothness Metric. Low = Transient, High = AGN.
            'fade_shape_correlation': fade_shape,    # Monotonicity of the tail.
            'baseline_ratio': baseline_ratio,        # History check (Pre-explosion brightness).
            'color_cooling_rate': color_cooling_rate,# Post-peak cooling rate.
            'color_slope_gr': color_slope_gr,        # Linear fit of cooling.

            # Color
            'ug_peak': ug_peak,  # UV Excess at peak.
            'gr_peak': gr_peak,  # Optical color at peak.
            'ur_peak': ur_peak,  # Wide-band color at peak.

            # linear ratios
            'flux_ratio_ug': flux_ratio_ug,
            'flux_ratio_gr': flux_ratio_gr,

            # distribution Stats 
            'flux_kurtosis': flux_kurtosis,  # "Spikiness" of flux distribution.
            'flux_skew': flux_skew,         # Asymmetry of flux distribution.

            'robust_duration': robust_duration,  # Time between 10th and 90th percentile.
            'duty_cycle': duty_cycle            # % of survey time active.
        }

    def process_object(self, lc_df, z, z_err, ebv):
        """
        Does all processing for a given lightcurve of an object
        """
        lc_df['EBV'] = float(ebv) 
        lc_clean = self.apply_deextinction(lc_df, log_df=None)
        lc_clean = self.apply_quality_cuts(lc_clean)

        # GP extraction returns dict, convert to df for model
        obj_id = lc_clean['object_id'].iloc[0]
        feature_dict = self.get_gp_features(obj_id, lc_clean)

        if feature_dict is None:
            print(f"GP Extraction failed for {obj_id}")
            return None

        features = pd.DataFrame([feature_dict]).fillna(0)
        
        safe_z = max(float(z), 0.0)
        dilation = 1.0 + safe_z

        features['redshift'] = safe_z
        features['redshift_err'] = float(z_err) 
        features['rest_rise_time'] = features['rise_time'] / dilation
        features['rest_fade_time'] = features['fade_time'] / dilation
        features['rest_fwhm'] = features['fwhm'] / dilation

        # amplitude correction
        safe_flux = features['amplitude'].clip(lower=0.001)
        features['absolute_magnitude_proxy'] = -2.5 * np.log10(safe_flux) - 5 * np.log10(z + 0.001)

        features['total_radiated_energy'] = features['total_radiated_energy_proxy'] * (z + 0.001)**2
        features['log_tde_error'] = np.log10(features['tde_power_law_error'] + 1e-9)

        return features.fillna(0)


    def predict_object(self, model, X):
        """
        Uses trained ensemble to generate prediction from features. 
        """
        p_base = model.models['base'].predict_proba(X)[:, 1]
        p_morph, p_phys = p_base 

        if 'morphology' in model.models:
            cols = [c for c in MORPHOLOGY_FEATURES if c in X.columns]
            p_morph = model.models['morphology'].predict_proba(X[cols])[:, 1]
            
        if 'physics' in model.models:
            cols = [c for c in PHYSICS_FEATURES if c in X.columns]
            p_phys = model.models['physics'].predict_proba(X[cols])[:, 1]

        p_mlp = model.models['mlp'].predict_proba(X)[:, 1]
        p_knn = model.models['knn'].predict_proba(X)[:, 1]

        final_prob = (0.48 * p_base) + (0.16 * p_morph) + (0.16 * p_phys) + (0.10 * p_mlp) + (0.10 * p_knn)

        return float(final_prob[0])

    # -------------------------
    #  END OF PROCESSING IMPL.
    # -------------------------

    def process(self, compound : T1Document, datapoints : Sequence[DataPoint], t2_views : Sequence[T2DocView]):

        # cannot process data if model is not loaded correctly. 
        if self.model is None: 
            return {"status" : "model missing"}

        
        z, z_err, ebv = 0.0, 0.0, 0.0
        
        for t2_view in t2_views: 
            # get ampel payload
            payload = t2_view.get_payload()
            if isinstance(payload, list):
                res = payload[-1] if payload else {}
            else:
                res = payload or {}

            if t2_view.unit == "T2DigestRedshifts": 
                z = res.get('ampel_z', 0.0)
                z_err = res.get('group_z_precision', 0.0)
            elif t2_view.unit in ['T2DestEchoEval', 'T2CatalogMatch']:
                if 'ebv' in res: 
                    ebv = res['ebv']


        # convert magnitude to flux: 
        # (I am unsure on whether this is the correct way to do this)
        # flux = 10 ** ((zp - mag) / 2.5)
        
        zp = 23.9 # might need to be changed

        records = []
        stock_id = compound.get('stock') if isinstance(compound, dict) else getattr(compound, 'stock', 'unknown')

        # map integer FIDs to bands (1=g, 2=r, 3=i for ZTF)
        FID_TO_BAND = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y', 6: 'u'}

        for dp in datapoints:
            dp_id = dp.get('id') if isinstance(dp, dict) else getattr(dp, 'id', None)
            
            # Skip invalid/upper-limit datapoints (keeps data clean like Kaggle)
            if dp_id is None or dp_id <= 0:
                continue

            b = dp.get('body', {}) if isinstance(dp, dict) else getattr(dp, 'body', {})
            
            # Determine filter and time
            fid = b.get('fid', 1)
            band = b.get('band', b.get('filter', FID_TO_BAND.get(fid, 'g')))
            mjd = b.get('mjd', b.get('jd', 0.0))
            
            if mjd == 0.0:
                continue

            if 'flux' in b and 'fluxErr' in b:
                # if ampel payload contains flux already
                flux = float(b['flux'])
                flux_err = float(b['fluxErr'])
                
            elif 'magpsf' in b and 'sigmapsf' in b:
                # if provided AB Magnitude
                mag = b['magpsf']
                mag_err = b['sigmapsf']
                
                flux = 10 ** ((zp - mag) / 2.5)
                
                # error propagation derivative
                # -> derivative of 10^x is 10^x * ln(10)
                #    -> ln(10) / 2.5 is 0.92103403719
                flux_err = flux * 0.92103403719 * mag_err
                
            else:
                # missing photometry data, skip
                continue

            # append to record
            records.append({
                'object_id': stock_id,
                'Time (MJD)': float(mjd),
                'Flux': float(flux),
                'Flux_err': float(flux_err),
                'Filter': band
            })

        if not records:
            return {"status": "no_photopoints", "is_tde": False, "tde_probability": 0.0}


        # construct lightcurve
        lc_df = pd.Dataframe(records)

        # process lc 
        features = self.process_object()
        
        # generate prediction
        pred, prob = 0, 0.0

        if features is None: 
            if getattr(self, "logger", None):
                self.logger.info(f"Error generating prediction, something is wrong with TDE Ensemble model.")
            
            return {
            "status": "failed to create features for object",
            "tde_probability": prob,
            "is_tde": pred,
            "features": features.to_dict(orient="records")[0]
            }

        else: 
            # create prediction from features: 
            X = features.drop(columns=['object_id'])
            prob = self.predict_object(self.model, X)
            pred = int(prob >= self.threshold)

            # debug
            if getattr(self, "logger", None):
                self.logger.info(f"Predicted {pred}, with probability {prob:.3f}")

        return {
        "status": "success",
        "tde_probability": prob,
        "is_tde": pred,
        "features": features.to_dict(orient="records")[0]
        }