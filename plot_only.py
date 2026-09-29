import matplotlib.pyplot as plt
import numpy as np

# Extracted data from your terminal output
# Arrays are ordered by splits: [10%, 20%, 40%]
means = {
    "ms": [np.mean([74.24, 86.55, 75.00]), np.mean([81.98, 86.93, 79.31]), np.mean([81.85, 89.72, 79.57])],
    "hhar": [np.mean([53.12, 54.21, 53.65]), np.mean([56.61, 52.19, 56.53]), np.mean([58.86, 54.94, 56.41])],
    "pamap2": [np.mean([47.33, 54.04, 47.73]), np.mean([48.45, 53.70, 48.28]), np.mean([49.84, 51.36, 49.10])],
    "ucihar": [np.mean([96.91, 96.07, 96.35]), np.mean([95.79, 96.35, 96.35]), np.mean([95.51, 96.35, 98.88])]
}

stds = {
    "ms": [np.std([74.24, 86.55, 75.00], ddof=1), np.std([81.98, 86.93, 79.31], ddof=1), np.std([81.85, 89.72, 79.57], ddof=1)],
    "hhar": [np.std([53.12, 54.21, 53.65], ddof=1), np.std([56.61, 52.19, 56.53], ddof=1), np.std([58.86, 54.94, 56.41], ddof=1)],
    "pamap2": [np.std([47.33, 54.04, 47.73], ddof=1), np.std([48.45, 53.70, 48.28], ddof=1), np.std([49.84, 51.36, 49.10], ddof=1)],
    "ucihar": [np.std([96.91, 96.07, 96.35], ddof=1), np.std([95.79, 96.35, 96.35], ddof=1), np.std([95.51, 96.35, 98.88], ddof=1)]
}

# Plotting Configuration
datasets_display = ["MS Dataset", "HHAR", "PAMAP2", "UCI HAR"]
dataset_keys = ["ms", "hhar", "pamap2", "ucihar"]
fractions = [0.10, 0.20, 0.40]

x = np.arange(len(fractions))
width = 0.2  
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
hatches = ['//', '\\\\', '--', 'xx']

fig, ax = plt.subplots(figsize=(10, 6), edgecolor='black', linewidth=1)

# Generate bars and error bars
for i, ds in enumerate(dataset_keys):
    bars = ax.bar(
        x + (i - 1.5) * width, 
        means[ds], 
        width, 
        label=datasets_display[i],
        color=colors[i], 
        hatch=hatches[i], 
        edgecolor='black',
        yerr=stds[ds], 
        capsize=4,
        error_kw={'elinewidth': 1, 'capthick': 1}
    )
    
    # Add accuracy text above bars
    for bar, err in zip(bars, stds[ds]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + err + 0.5, 
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold'
        )

# Styling
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Data Percentage', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['10%', '20%', '40%'], fontsize=12, fontweight='bold')
ax.set_ylim(0, 115) # Provides extra headroom for text

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Datasets', title_fontsize=12, fontsize=10, loc='upper center', ncol=4, framealpha=1)

plt.tight_layout()
plt.savefig('combined_accuracy_plot_realistic.png', dpi=300, bbox_inches='tight')
print("\nPlot saved successfully as 'combined_accuracy_plot_realistic.png'!")