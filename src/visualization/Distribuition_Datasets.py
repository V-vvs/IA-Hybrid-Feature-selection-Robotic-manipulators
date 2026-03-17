import pandas as pd
import ast
import plotly.express as px
import plotly.io as pio

# Function to load and process a single LP dataset
def load_and_process_lp_dataset(filename, dataset_name):
    df = pd.read_csv(filename)

    # Normalize 'label' column to 'class'
    class_col = [col for col in df.columns if col.lower() in ['class', 'label']]
    if class_col:
        df.rename(columns={class_col[0]: 'class'}, inplace=True)
    else:
        raise ValueError(f"No class/label column found in {filename}")

    # Convert stringified time series to actual lists
    for col in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
            raise ValueError(f"Missing expected column '{col}' in {filename}")

    # Remove samples where all 6 time series are identical
    def is_duplicate(row):
        s = row[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']]
        return all(s.iloc[0] == val for val in s)

    df = df[~df.apply(is_duplicate, axis=1)]
    df['source'] = dataset_name
    return df

# Load and process all LP datasets
lp_datasets = []
for i in range(1, 6):
    filename = f'data/raw/transformado_lp{i}.csv'
    dataset = load_and_process_lp_dataset(filename, f'LP{i}')
    lp_datasets.append(dataset)

# Combine all LP datasets
lp_combined = pd.concat(lp_datasets, ignore_index=True)

# Friendly name mapping
mapping_dict = {
    'front_col': 'front collision',
    'fr_collision': 'front collision',
    'left_col': 'left collision',
    'back_col': 'back collision',
    'right_col': 'right collision',
    'bottom_collision': 'bottom collision',
    'bottom_obstruction': 'bottom obstruction',
    'collision_in_part': 'collision in part',
    'collision_in_tool': 'collision in tool',
    'slightly_moved': 'slightly moved'
}
lp_combined['class'] = lp_combined['class'].replace(mapping_dict)

# Group for stacked bar
class_distribution = lp_combined.groupby(['class', 'source']).size().reset_index(name='count')

# Totals for annotations
totals = class_distribution.groupby('class')['count'].sum().reset_index()

# Ordering
class_order = totals.sort_values('count', ascending=False)['class']
class_distribution['class'] = pd.Categorical(class_distribution['class'], categories=class_order, ordered=True)

# Plot stacked bar
fig_lp = px.bar(
    class_distribution,
    x='class',
    y='count',
    color='source',
    labels={'class': 'Class', 'count': 'Sample Count', 'source': 'Dataset'}
)

# Layout adjustments for fig_lp
fig_lp.update_layout(
    #title='Class Distribution Across LP Datasets',
    xaxis_title='Class',
    yaxis_title='Sample Count',
    xaxis_title_font=dict(size=30),
    yaxis_title_font=dict(size=30),
    legend_title_font=dict(size=30),
    legend_font=dict(size=30),
    xaxis_tickfont=dict(size=34),
    yaxis_tickfont=dict(size=30),
    xaxis={'categoryorder': 'array', 'categoryarray': class_order}
)

# Add annotations
for _, row in totals.iterrows():
    fig_lp.add_annotation(
        x=row['class'],
        y=row['count'],
        text=str(row['count']),
        showarrow=False,
        font=dict(size=28, color='black'),
        yshift=18
    )

# Save and show
pio.write_html(fig_lp, file='lp_class_distribution.html', auto_open=False)
fig_lp.show()

# Load and process dataset_robot.csv
robot_df = pd.read_csv('data/raw/dataset_robot.csv')

# Normalize 'label' column to 'class'
class_col = [col for col in robot_df.columns if col.lower() in ['class', 'label']]
if class_col:
    robot_df.rename(columns={class_col[0]: 'class'}, inplace=True)
else:
    raise ValueError("No class/label column found in dataset_robot.csv")

# Group and sort
robot_class_counts = robot_df['class'].value_counts().reset_index()
robot_class_counts.columns = ['class', 'count']
robot_class_counts = robot_class_counts.sort_values('count', ascending=False)
class_order_robot = robot_class_counts['class']
robot_class_counts['class'] = pd.Categorical(robot_class_counts['class'], categories=class_order_robot, ordered=True)

# Plot robot
fig_robot = px.bar(
    robot_class_counts,
    x='class',
    y='count',
    labels={'class': 'Class', 'count': 'Sample Count'},
    color_discrete_sequence=['steelblue']
)

# Layout adjustments for fig_robot
fig_robot.update_traces(width=0.4)
fig_robot.update_layout(
    #title='Class Distribution - Robot Dataset',
    xaxis_title='Class',
    yaxis_title='Sample Count',
    xaxis_title_font=dict(size=30),
    yaxis_title_font=dict(size=30),
    legend_title_font=dict(size=30),
    legend_font=dict(size=30),
    bargap=0.5,
    xaxis_tickfont=dict(size=34),
    yaxis_tickfont=dict(size=30),
    xaxis={'categoryorder': 'array', 'categoryarray': class_order_robot}
)

# Annotations
for _, row in robot_class_counts.iterrows():
    fig_robot.add_annotation(
        x=row['class'],
        y=row['count'],
        text=str(row['count']),
        showarrow=False,
        font=dict(size=28, color='black'),
        yshift=22
    )

# Save and show
pio.write_html(fig_robot, file='robot_class_distribution.html', auto_open=False)
fig_robot.show()
