import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Needed for rolling mean

def visualize_cum_rewards(cum_rewards, additional_info='', smooth_rate=1):
    # Set the style of seaborn for more sophisticated visuals
    sns.set(style='whitegrid')

    # Create a plot with larger size for better visibility
    plt.figure(figsize=(10, 6))

    # Checking if smoothing is required
    if smooth_rate > 1:
        # Converting list to DataFrame for rolling operation
        cum_rewards_df = pd.DataFrame(cum_rewards, columns=['Cumulative Reward'])
        # Calculating rolling mean
        smooth_data = cum_rewards_df.rolling(window=smooth_rate, center=True).mean()
    else:
        smooth_data = pd.DataFrame(cum_rewards, columns=['Cumulative Reward'])

    # Plotting the cumulative rewards with or without smoothing
    sns.lineplot(data=smooth_data, color='blue', linewidth=2.5)

    # Setting labels and title with enhanced font properties
    plt.xlabel('Episodes', fontsize=14, labelpad=15)
    plt.ylabel('Cumulative Reward', fontsize=14, labelpad=15)
    plt.title('Cumulative Reward vs Episodes', fontsize=16, pad=20)

    # Enhancing tick parameters for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    os.makedirs('figures', exist_ok=True)
    # Saving the plot with high resolution
    plt.savefig(f'figures/cumulative_reward_{additional_info}_sm{smooth_rate}.png', dpi=300)

    # Closing the plot to free up memory
    plt.close()