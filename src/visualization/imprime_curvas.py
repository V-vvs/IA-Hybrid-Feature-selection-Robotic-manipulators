import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

# -------------------- Function Definition: parse_string_list -------------- #
def parse_string_list(s):
    try:
        if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
            s_corrected = s.replace("'", '"')
            s_corrected = s_corrected.replace("nan", "null").replace("NaN", "null")
            s_corrected = s_corrected.replace("inf", '"Infinity"').replace("-inf", '"-Infinity"')
            parsed_list = json.loads(s_corrected)
            for i, item in enumerate(parsed_list):
                if item == "Infinity": parsed_list[i] = np.inf
                elif item == "-Infinity": parsed_list[i] = -np.inf
                elif item is None: parsed_list[i] = np.nan
            return np.array(parsed_list, dtype=float)
        elif isinstance(s, (list, np.ndarray)):
            return np.array(s, dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return np.array([])

# -------------------- Configuration and Data Loading -------------- #
output_dir = 'TimeSeries_Plots_V11_EN'
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv('data/raw/dataset_robot.csv')
except FileNotFoundError:
    print("Error: File 'dataset_robot.csv' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

print(f"Original dataset shape: {df.shape}")

time_series_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
for col in time_series_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_string_list)
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Creating empty column.")
        df[col] = pd.Series([np.array([]) for _ in range(len(df))])

df.dropna(subset=time_series_cols, how='all', inplace=True)
if time_series_cols[0] in df.columns:
    df = df[df[time_series_cols].applymap(lambda x: x.size > 0 if isinstance(x, np.ndarray) else False).any(axis=1)]
else:
    print(f"Critical column {time_series_cols[0]} is missing, cannot proceed.")
    exit()

if 'label' not in df.columns:
    print("Error: Column 'label' not found. Exiting.")
    exit()
if df.empty:
    print("Error: DataFrame is empty after loading and initial processing. Cannot proceed.")
    exit()
df['label'] = df['label'].astype(str)
print(f"Dataset shape after parsing and cleaning: {df.shape}")

def plot_individual_curves(df_data, force_cols, torque_cols, output_name_base):
    """
    Plot individual time series curves without envelope analysis
    """
    labels = sorted(df_data['label'].unique())
    num_labels = len(labels)

    if num_labels == 0:
        print("No labels found after data processing. Cannot generate plot.")
        return

    subplot_titles_row1 = [f"{label}" for label in labels]
    subplot_titles_row2 = ["" for label in labels]
    final_subplot_titles = subplot_titles_row1 + subplot_titles_row2

    fig = make_subplots(
        rows=2, cols=num_labels,
        subplot_titles=final_subplot_titles,
        shared_xaxes=False,
        shared_yaxes='rows',
        vertical_spacing=0.2,
        horizontal_spacing=0.08
    )

    # Improved color scheme
    colors = px.colors.qualitative.Set1
    axis_colors = {
        'x': colors[0],  # Blue
        'y': colors[1],  # Orange  
        'z': colors[2]   # Green
    }
    
    def get_axis_color(var_name):
        suffix = var_name.lower()[-1]
        return axis_colors.get(suffix, colors[3])

    legend_items_added = set()

    for col_idx_plot, label_val in enumerate(labels):
        df_label_subset = df_data[df_data['label'] == label_val]
            
        print(f"Class {label_val}: showing {len(df_label_subset)} samples")

        for _, sample_row in df_label_subset.iterrows():
            # Plot Forces
            for var_f in force_cols:
                force_data = sample_row.get(var_f)
                if isinstance(force_data, np.ndarray) and force_data.size > 0:
                    time_steps = np.arange(len(force_data))
                    show_legend_f = (f"{var_f}" not in legend_items_added)
                    fig.add_trace(
                        go.Scatter(
                            x=time_steps, y=force_data, mode='lines', name=f"{var_f}",
                            legendgroup=f"{var_f}",
                            line=dict(width=1.2, color=get_axis_color(var_f)),
                            showlegend=show_legend_f, 
                            opacity=0.65,
                            hovertemplate=f"{var_f}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                        ), row=1, col=col_idx_plot + 1
                    )
                    if show_legend_f: legend_items_added.add(f"{var_f}")

            # Plot Torques
            for var_t in torque_cols:
                torque_data = sample_row.get(var_t)
                if isinstance(torque_data, np.ndarray) and torque_data.size > 0:
                    time_steps = np.arange(len(torque_data))
                    show_legend_t = (f"{var_t}" not in legend_items_added)
                    fig.add_trace(
                        go.Scatter(
                            x=time_steps, y=torque_data, mode='lines', name=f"{var_t}",
                            legendgroup=f"{var_t}",
                            line=dict(width=1.2, color=get_axis_color(var_t)),
                            showlegend=show_legend_t, 
                            opacity=0.65,
                            hovertemplate=f"{var_t}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                        ), row=2, col=col_idx_plot + 1
                    )
                    if show_legend_t: legend_items_added.add(f"{var_t}")

    fig.update_layout(
        height=1000,
        width=max(1400, 450 * num_labels + 350),
        title_text="Force and Torque Time Series by Class - Individual Curves",
        title_x=0.5,
        title_font_size=22,
        showlegend=True,
        legend=dict(
            title_text='Variables',
            title_font_size=24,
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=21)
        ),
        margin=dict(l=140, r=250, t=120, b=100),
        plot_bgcolor='rgba(248,248,248,1)',
        paper_bgcolor='white'
    )

    # Update subplot titles font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=24)

    # Update axes with grid and zero line
    for c_idx in range(1, num_labels + 1):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', 
                        title_text="", showticklabels=True, row=1, col=c_idx,
                        tickfont=dict(size=18))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        title_text="Time Step", row=2, col=c_idx, 
                        title_standoff=15, title_font_size=24,
                        tickfont=dict(size=18))
        
        # Add zero reference line and grid
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        zeroline=True, zerolinewidth=2, zerolinecolor='black',
                        row=1, col=c_idx, tickfont=dict(size=18))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        zeroline=True, zerolinewidth=2, zerolinecolor='black',
                        row=2, col=c_idx, tickfont=dict(size=18))

    # Y-axis titles
    fig.update_yaxes(
        title_text="Force (N)",
        row=1, col=1,
        title_standoff=23, 
        title_font=dict(size=23, color="black"),
    )
    fig.update_yaxes(
        title_text="Torque (N⋅m)",
        row=2, col=1,
        title_standoff=23,
        title_font=dict(size=23, color="black"),
    )

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.write_image(os.path.join(output_dir, f"{output_name_base}.png"), scale=3.0)
        fig.write_html(os.path.join(output_dir, f"{output_name_base}.html"))
        print(f"Figure saved to: {output_dir}/{output_name_base}.png and .html")
    except Exception as e:
        print(f"Error saving figure: {e}")

def plot_envelope_analysis(df_data, force_cols, torque_cols, output_name_base):
    """
    Plot envelope analysis with median, percentiles, and min/max bounds
    """
    labels = sorted(df_data['label'].unique())
    num_labels = len(labels)

    if num_labels == 0:
        print("No labels found after data processing. Cannot generate plot.")
        return

    subplot_titles_row1 = [f"{label}" for label in labels]
    subplot_titles_row2 = ["" for label in labels]
    final_subplot_titles = subplot_titles_row1 + subplot_titles_row2

    fig = make_subplots(
        rows=2, cols=num_labels,
        subplot_titles=final_subplot_titles,
        shared_xaxes=False,
        shared_yaxes='rows',
        vertical_spacing=0.2,
        horizontal_spacing=0.08
    )

    colors = px.colors.qualitative.Set1
    axis_colors = {
        'x': colors[0],
        'y': colors[1],  
        'z': colors[2]
    }
    
    def get_axis_color(var_name):
        suffix = var_name.lower()[-1]
        return axis_colors.get(suffix, colors[3])

    legend_items_added = set()

    for col_idx_plot, label_val in enumerate(labels):
        df_label_subset = df_data[df_data['label'] == label_val]
        
        # Collect data
        force_data_dict = {col: [] for col in force_cols}
        torque_data_dict = {col: [] for col in torque_cols}
        
        for _, sample_row in df_label_subset.iterrows():
            for var_f in force_cols:
                force_data = sample_row.get(var_f)
                if isinstance(force_data, np.ndarray) and force_data.size > 0:
                    force_data_dict[var_f].append(force_data)
                    
            for var_t in torque_cols:
                torque_data = sample_row.get(var_t)
                if isinstance(torque_data, np.ndarray) and torque_data.size > 0:
                    torque_data_dict[var_t].append(torque_data)

        # Plot force envelopes
        for var_f in force_cols:
            if force_data_dict[var_f]:
                max_len = max(len(arr) for arr in force_data_dict[var_f])
                
                interpolated_data = []
                for arr in force_data_dict[var_f]:
                    if len(arr) < max_len:
                        x_old = np.linspace(0, 1, len(arr))
                        x_new = np.linspace(0, 1, max_len)
                        arr_interp = np.interp(x_new, x_old, arr)
                        interpolated_data.append(arr_interp)
                    else:
                        interpolated_data.append(arr[:max_len])
                
                data_matrix = np.array(interpolated_data)
                median_data = np.median(data_matrix, axis=0)
                min_data = np.min(data_matrix, axis=0)
                max_data = np.max(data_matrix, axis=0)
                percentile_25 = np.percentile(data_matrix, 25, axis=0)
                percentile_75 = np.percentile(data_matrix, 75, axis=0)
                time_steps = np.arange(max_len)
                
                show_legend = (f"{var_f}" not in legend_items_added)
                
                # Min-Max envelope (light)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_steps, time_steps[::-1]]),
                        y=np.concatenate([max_data, min_data[::-1]]),
                        fill='toself',
                        fillcolor=get_axis_color(var_f).replace('rgb', 'rgba').replace(')', ', 0.15)'),
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{var_f}_minmax"
                    ), row=1, col=col_idx_plot + 1
                )
                
                # 25-75 percentile band (darker)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_steps, time_steps[::-1]]),
                        y=np.concatenate([percentile_75, percentile_25[::-1]]),
                        fill='toself',
                        fillcolor=get_axis_color(var_f).replace('rgb', 'rgba').replace(')', ', 0.35)'),
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{var_f}_p25p75"
                    ), row=1, col=col_idx_plot + 1
                )
                
                # Median line
                fig.add_trace(
                    go.Scatter(
                        x=time_steps, y=median_data, mode='lines', 
                        name=f"{var_f}",
                        line=dict(width=4, color=get_axis_color(var_f)),
                        showlegend=show_legend,
                        hovertemplate=f"{var_f}<br>Time: %{{x}}<br>Median: %{{y:.3f}}<extra></extra>"
                    ), row=1, col=col_idx_plot + 1
                )
                if show_legend: legend_items_added.add(f"{var_f}")

        # Plot torque envelopes
        for var_t in torque_cols:
            if torque_data_dict[var_t]:
                max_len = max(len(arr) for arr in torque_data_dict[var_t])
                
                interpolated_data = []
                for arr in torque_data_dict[var_t]:
                    if len(arr) < max_len:
                        x_old = np.linspace(0, 1, len(arr))
                        x_new = np.linspace(0, 1, max_len)
                        arr_interp = np.interp(x_new, x_old, arr)
                        interpolated_data.append(arr_interp)
                    else:
                        interpolated_data.append(arr[:max_len])
                
                data_matrix = np.array(interpolated_data)
                median_data = np.median(data_matrix, axis=0)
                min_data = np.min(data_matrix, axis=0)
                max_data = np.max(data_matrix, axis=0)
                percentile_25 = np.percentile(data_matrix, 25, axis=0)
                percentile_75 = np.percentile(data_matrix, 75, axis=0)
                time_steps = np.arange(max_len)
                
                show_legend = (f"{var_t}" not in legend_items_added)
                
                # Min-Max envelope (light)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_steps, time_steps[::-1]]),
                        y=np.concatenate([max_data, min_data[::-1]]),
                        fill='toself',
                        fillcolor=get_axis_color(var_t).replace('rgb', 'rgba').replace(')', ', 0.15)'),
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{var_t}_minmax"
                    ), row=2, col=col_idx_plot + 1
                )
                
                # 25-75 percentile band (darker)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_steps, time_steps[::-1]]),
                        y=np.concatenate([percentile_75, percentile_25[::-1]]),
                        fill='toself',
                        fillcolor=get_axis_color(var_t).replace('rgb', 'rgba').replace(')', ', 0.35)'),
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{var_t}_p25p75"
                    ), row=2, col=col_idx_plot + 1
                )
                
                # Median line
                fig.add_trace(
                    go.Scatter(
                        x=time_steps, y=median_data, mode='lines', 
                        name=f"{var_t}",
                        line=dict(width=4, color=get_axis_color(var_t)),
                        showlegend=show_legend,
                        hovertemplate=f"{var_t}<br>Time: %{{x}}<br>Median: %{{y:.3f}}<extra></extra>"
                    ), row=2, col=col_idx_plot + 1
                )
                if show_legend: legend_items_added.add(f"{var_t}")

    fig.update_layout(
        height=1000,
        width=max(1400, 450 * num_labels + 350),
        title_text="Force and Torque Envelope Analysis (Median + Min/Max + IQR)",
        title_x=0.5,
        title_font_size=22,
        showlegend=True,
        legend=dict(
            title_text='Variables (Median)',
            title_font_size=24,
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=21)
        ),
        margin=dict(l=140, r=280, t=120, b=100),
        plot_bgcolor='rgba(248,248,248,1)',
        paper_bgcolor='white'
    )

    # Update subplot titles font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=24)

    # Update axes with grid and zero line
    for c_idx in range(1, num_labels + 1):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', 
                        title_text="", showticklabels=True, row=1, col=c_idx,
                        tickfont=dict(size=14))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        title_text="Time Step", row=2, col=c_idx, 
                        title_standoff=15, title_font_size=24,
                        tickfont=dict(size=14))
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        zeroline=True, zerolinewidth=2, zerolinecolor='black',
                        row=1, col=c_idx, tickfont=dict(size=14))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        zeroline=True, zerolinewidth=2, zerolinecolor='black',
                        row=2, col=c_idx, tickfont=dict(size=14))

    fig.update_yaxes(
        title_text="Force (N)",
        row=1, col=1,
        title_standoff=23, 
        title_font=dict(size=23, color="black"),
    )
    fig.update_yaxes(
        title_text="Torque (N⋅m)",
        row=2, col=1,
        title_standoff=23,
        title_font=dict(size=23, color="black"),
    )

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.write_image(os.path.join(output_dir, f"{output_name_base}.png"), scale=3.0)
        fig.write_html(os.path.join(output_dir, f"{output_name_base}.html"))
        print(f"Figure saved to: {output_dir}/{output_name_base}.png and .html")
    except Exception as e:
        print(f"Error saving figure: {e}")

# Execute plotting if data is available
if not df.empty:
    # Option 1: Individual curves without envelope
    plot_individual_curves(
        df,
        force_cols=['Fx', 'Fy', 'Fz'],
        torque_cols=['Tx', 'Ty', 'Tz'],
        output_name_base='ForceTorque_Individual_Curves'
    )
    
    # Option 2: Envelope analysis (median + percentiles + min/max)
    plot_envelope_analysis(
        df,
        force_cols=['Fx', 'Fy', 'Fz'],
        torque_cols=['Tx', 'Ty', 'Tz'],
        output_name_base='ForceTorque_Envelope_Analysis'
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION OPTIONS GENERATED:")
    print("="*60)
    print("1. Individual Curves - Shows all raw time series (ALL SAMPLES)")
    print("2. Envelope Analysis - Median + percentiles + min/max bounds (ALL SAMPLES)")
    print("="*60)
else:
    print("DataFrame is empty after final preprocessing. Cannot generate plots.")