import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file (replace with the actual name of your file)
df = pd.read_csv("view_and_graph_view.csv")

# Filter only data where Testbed == 'Fabric'
df_fabric = df[df["Testbed"] == "Fabric"]

# Manually calculate the average to ensure correct values
grouped = df_fabric.groupby(["Algorithm", "Operation"])["MAPE"].mean().reset_index()

# Creating the Interval Plot for the MAPE metric, differentiating Read and Write operations
plt.figure(figsize=(11, 7))
ax = sns.pointplot(
    x="Algorithm", 
    y="MAPE", 
    hue="Operation",  
    data=df_fabric, 
    ci="sd", 
    capsize=0.2, 
    markers="o", 
    errwidth=1
)

# Setting the colors of the boxes for each operation
colors = {
    "Read": {"text": "blue", "box": "white"},
    "Write": {"text": "orange", "box": "white"}
}

# Adding the average numbers on the points with fine-tuning the first point
for i, row in grouped.iterrows():
    x = list(df_fabric["Algorithm"].unique()).index(row["Algorithm"])  # Get the correct position on the X-axis
    y = row["MAPE"]  # Get the average of MAPE
    
    operation = row["Operation"]  # Get the operation (Read or Write)
    text_color = colors[operation]["text"]
    box_color = colors[operation]["box"]

    # Fine-tuning for the first point
    y_offset = 0.7 if i == 0 else 0  # Slightly shifts the first point down
    
    plt.text(
        x, y + y_offset, f"{y:.2f}", 
        ha="center", va="bottom", fontsize=16, color=text_color, 
       bbox=dict(facecolor=box_color, edgecolor=box_color, alpha=0.7, pad=1)
    )

# Customizing the plot
plt.xlabel("Algorithm", fontsize=16)
plt.ylabel("MAPE", fontsize=16)
plt.title("MAPE by Algorithm in Fabric", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adjusting the legend to remove the outline
legend = plt.legend(title="Operation", fontsize=16, title_fontsize=16, frameon=False)

# Remove vertical grid and keep only horizontal
ax.grid(True, which='major', axis='y', linestyle='--')
ax.grid(False, which='major', axis='x')

# Save the plot
plt.savefig("mape_fabric_basic_algorithms.pdf")  # Also save as PDF