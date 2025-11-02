import json

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 10 (index 18) - Visualization Pipeline with subplots
cell10 = nb['cells'][18]
cell10['source'] = [
    "print(\"\\n📊 STEP 9: Visualization Pipeline\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "from PIL import Image\n",
    "\n",
    "# Generate SHAP summary plot\n",
    "shap_plot_path = visualization.plot_shap_summary(\n",
    "    explanation.shap_values,\n",
    "    explanation.feature_names,\n",
    "    output_name=\"pipeline_demo_shap\"\n",
    ")\n",
    "\n",
    "# Generate feature importance plot\n",
    "importance_plot_path = explainability_viz.plot_feature_importance(\n",
    "    explanation.feature_names,\n",
    "    explanation.shap_values,\n",
    "    output_name=\"pipeline_demo_importance\"\n",
    ")\n",
    "\n",
    "# Display in subplots\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 6))\n",
    "\n",
    "img1 = Image.open(shap_plot_path)\n",
    "axes[0].imshow(img1)\n",
    "axes[0].axis('off')\n",
    "axes[0].set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')\n",
    "\n",
    "img2 = Image.open(importance_plot_path)\n",
    "axes[1].imshow(img2)\n",
    "axes[1].axis('off')\n",
    "axes[1].set_title('Feature Importance Bar Chart', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\n✓ SHAP plot saved: {shap_plot_path}\")\n",
    "print(f\"✓ Feature importance plot saved: {importance_plot_path}\")"
]

# Update Cell 11 (index 20) - Advanced Visualization with pairplot
cell11 = nb['cells'][20]
cell11['source'] = [
    "print(\"\\n📈 STEP 10: Advanced Visualization\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "from PIL import Image\n",
    "\n",
    "# Create pairplot\n",
    "pairplot_path = visualization_engine.pairplot(\n",
    "    training_df,\n",
    "    hue_column=\"risk_label\",\n",
    "    output_name=\"pipeline_demo_pairplot\"\n",
    ")\n",
    "print(f\"\\nPairplot saved: {pairplot_path}\")\n",
    "img = Image.open(pairplot_path)\n",
    "plt.figure(figsize=(14, 10))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.title('Feature Pairplot by Risk Level', fontsize=16, fontweight='bold', pad=20)\n",
    "plt.show()\n",
    "\n",
    "# Create correlation heatmap and histogram in subplots\n",
    "heatmap_path = visualization_engine.heatmap(\n",
    "    training_df,\n",
    "    output_name=\"pipeline_demo_heatmap\"\n",
    ")\n",
    "\n",
    "hist_path = visualization_engine.histogram(\n",
    "    training_df,\n",
    "    column=\"credit_score\",\n",
    "    output_name=\"pipeline_demo_histogram\"\n",
    ")\n",
    "\n",
    "# Display in subplots\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 6))\n",
    "\n",
    "img1 = Image.open(heatmap_path)\n",
    "axes[0].imshow(img1)\n",
    "axes[0].axis('off')\n",
    "axes[0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')\n",
    "\n",
    "img2 = Image.open(hist_path)\n",
    "axes[1].imshow(img2)\n",
    "axes[1].axis('off')\n",
    "axes[1].set_title('Credit Score Distribution', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\n✓ Heatmap saved: {heatmap_path}\")\n",
    "print(f\"✓ Histogram saved: {hist_path}\")\n",
    "print(\"\\n✓ Advanced visualizations generated\")"
]

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Enhanced visualization cells updated!")
print("- Cell 10: Feature importance with subplots (2 charts side-by-side)")
print("- Cell 11: Pairplot + Heatmap/Histogram with subplots (3 charts total)")
