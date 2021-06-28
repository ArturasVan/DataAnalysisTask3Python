import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('adult.data.csv')

# Add 'overweight' column
df['overweight'] = df['weight'] / (df['height']/100)**2
def overweight(x):
    if x['overweight'] < 25:
        return 0
    else:
        return 1

df['overweight'] = df.apply(overweight, axis=1)
# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'].values[df['cholesterol'] <= 1] = 0
df['cholesterol'].values[df['cholesterol'] >1] = 1
df['gluc'].values[df['gluc'] <= 1] = 0
df['gluc'].values[df['gluc'] >1] = 1

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(
        frame=df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
        id_vars=['cardio']
    )


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.DataFrame(
        df_cat.groupby(
                ['variable', 'value', 'cardio'])['value'].count()).rename(
                columns={'value': 'total'}).reset_index()

    # Draw the catplot with 'sns.catplot()'
    sns.catplot(x='variable', y='total', data=df_cat, hue='value', col='cardio', kind='bar')


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df
    df_heat = df_heat[df_heat["ap_lo"] <= df_heat["ap_hi"]]
    df_heat = df_heat[df_heat["height"] >= df_heat["height"].quantile(0.025)]
    df_heat = df_heat[df_heat["height"] <= df_heat["height"].quantile(0.975)]
    df_heat = df_heat[df_heat["weight"] >= df_heat["weight"].quantile(0.025)]
    df_heat = df_heat[df_heat["weight"] <= df_heat["weight"].quantile(0.975)]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    
    sns.heatmap(corr, mask = mask, fmt='.1f', vmax = 0.3, center = 0, annot = True,
           square = True, linewidths=0.5, cbar_kws={'shrink': 0.45,'format':'%.2f'})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
