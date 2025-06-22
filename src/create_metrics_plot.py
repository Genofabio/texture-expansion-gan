import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Percorso alla cartella contenente i modelli
base_path = "../data/evaluation/extraset/"
save_dir = "../results/evaluation/extraset/"
os.makedirs(save_dir, exist_ok=True)

# ======= PLOT AGGREGATI =======

# Lista per raccogliere i dati aggregati
all_metrics = []

for model_folder in os.listdir(base_path):
    model_path = os.path.join(base_path, model_folder)
    if not os.path.isdir(model_path):
        continue

    metrics_file = os.path.join(model_path, "generated", f"{model_folder}_metrics_aggregated.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            metrics["model"] = model_folder
            all_metrics.append(metrics)

# DataFrame aggregato
df = pd.DataFrame(all_metrics)
print(df.columns)
df.set_index("model", inplace=True)

# Classifiche
print("📊 Classifica per PSNR (più alto è meglio):")
print(df.sort_values(by="psnr", ascending=False)[["psnr"]])
print("\n📊 Classifica per SSIM (più alto è meglio):")
print(df.sort_values(by="ssim", ascending=False)[["ssim"]])
print("\n📊 Classifica per LPIPS (più basso è meglio):")
print(df.sort_values(by="lpips", ascending=True)[["lpips"]])

# Plot aggregati
metrics_to_plot = ["psnr", "ssim", "lpips"]
fig, axes = plt.subplots(nrows=3, figsize=(10, 15))

for ax, metric in zip(axes, metrics_to_plot):
    ascending = (metric == "lpips")
    df_sorted = df.sort_values(by=metric, ascending=ascending)
    best_val = df_sorted[metric].min() if ascending else df_sorted[metric].max()
    colors = ['skyblue' if val != best_val else 'purple' for val in df_sorted[metric]]
    df_sorted[metric].plot(kind='barh', ax=ax, color=colors)
    ax.set_title(f"{metric.upper()}")
    ax.set_ylabel("")

fig.subplots_adjust(hspace=1.0)
fig.suptitle("Metriche aggregate su immagini usate nel training", fontsize=16)
agg_plot_path = os.path.join(save_dir, "aggregated_metrics.png")
plt.savefig(agg_plot_path)
plt.close()

# ======= PLOT DETTAGLIATI =======

# Raccolta dati dettagliati
detailed_all = []

for model_folder in os.listdir(base_path):
    model_path = os.path.join(base_path, model_folder)
    if not os.path.isdir(model_path):
        continue

    detailed_file = os.path.join(model_path, "generated", f"{model_folder}_metrics_detailed.json")
    if os.path.exists(detailed_file):
        with open(detailed_file, "r") as f:
            data = json.load(f)
            for d in data:
                d["model"] = model_folder
            detailed_all.extend(data)

# DataFrame dettagliato
df_detailed = pd.DataFrame(detailed_all)

# Boxplot per ogni metrica
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_detailed, x="model", y=metric)
    plt.title(f"Distribuzione {metric.upper()} per modello")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_path = os.path.join(save_dir, f"boxplot_{metric}.png")
    plt.savefig(boxplot_path)
    plt.close()

# Scatter plot PSNR (y) vs LPIPS (x)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_detailed, x="lpips", y="psnr", hue="model")
plt.title("PSNR vs LPIPS per immagine")
plt.xlabel("LPIPS (sinistra è meglio)")
plt.ylabel("PSNR (alto è meglio)")
plt.tight_layout()
scatter_path = os.path.join(save_dir, "scatter_psnr_vs_lpips.png")
plt.savefig(scatter_path)
plt.close()
