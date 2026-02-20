import matplotlib.pyplot as plt
import numpy as np
import os

def get_best_filter(df_obj):
    """Returns the filter (u,g,r,i,z,y) with the most data points."""
    if df_obj.empty: 
        return 'r'
    return df_obj['filter'].value_counts().idxmax()

# Now it accepts the dataframe and save directory directly
def generate_transient_plots(df, save_dir):
    # --- FIND EXEMPLARS ---
    def find_best_example(label_substrings, max_duration=None, min_duration=None, exclude_ids=[]):
        mask = df['SpecType'].astype(str).apply(lambda x: any(s.lower() in x.lower() for s in label_substrings))
        candidates = df[mask & (~df['object_id'].isin(exclude_ids))]
        
        if candidates.empty and (max_duration or min_duration):
            print(f"[!] No label match for {label_substrings}. Using duration fallback.")
            non_tde = df[(df['target'] == 0) & (~df['object_id'].isin(exclude_ids))]
            spans = non_tde.groupby('object_id')['mjd'].agg(np.ptp)
            
            if max_duration:
                valid = spans[(spans < max_duration) & (spans > 10)].index
                candidates = df[df['object_id'].isin(valid)]
            elif min_duration:
                valid = spans[spans > min_duration].index
                candidates = df[df['object_id'].isin(valid)]

        if candidates.empty:
            return None, "None"
        
        best_id = candidates['object_id'].value_counts().idxmax()
        try:
            actual_type = candidates[candidates['object_id'] == best_id]['SpecType'].iloc[0]
        except Exception:
            actual_type = "Unknown"
            
        return best_id, actual_type

    tde_id = df[df['target'] == 1]['object_id'].value_counts().idxmax()
    sn_id, sn_type = find_best_example(['SN', 'Supernova', 'Ia', 'Ib', 'II'], max_duration=100, exclude_ids=[tde_id])
    agn_id, agn_type = find_best_example(['AGN', 'QSO', 'Quasar'], min_duration=500, exclude_ids=[tde_id, sn_id])

    # --- PLOTTING ---
    objects = [
        ("TDE (Tidal Disruption)", tde_id, '#e74c3c'), 
        (f"Supernova ({sn_type})", sn_id, '#27ae60'), 
        (f"AGN ({agn_type})",      agn_id, '#34495e') 
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    for i, (label, obj_id, color) in enumerate(objects):
        ax = axes[i]
        if obj_id is None:
            ax.text(0.5, 0.5, f"No Data Found for {label}", ha='center')
            continue
            
        obj_data = df[df['object_id'] == obj_id]
        best_band = get_best_filter(obj_data)
        plot_data = obj_data[obj_data['filter'] == best_band].copy()
        
        peak_idx = plot_data['flux'].idxmax()
        t_peak = plot_data.loc[peak_idx, 'mjd']
        plot_data['mjd_norm'] = plot_data['mjd'] - t_peak
        
        ax.errorbar(plot_data['mjd_norm'], plot_data['flux'], yerr=plot_data['flux_err'], 
                    fmt='o', color=color, ecolor='silver', alpha=0.7, label=f'Flux ({best_band}-band)')
        
        plot_data = plot_data.sort_values('mjd_norm')
        ax.plot(plot_data['mjd_norm'], plot_data['flux'], color=color, alpha=0.3)

        ax.set_title(f"{label} | ID: {obj_id}", fontweight='bold', color=color)
        ax.set_ylabel("Flux ($\mu$Jy)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

        if "AGN" not in label and "Variable" not in label:
            ax.set_xlim(-100, 200)

    axes[2].set_xlabel("Days from Peak")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "transient_comparison_robust.png")
    plt.savefig(save_path, dpi=300)
    print(f"   [+] Saved {os.path.basename(save_path)}")