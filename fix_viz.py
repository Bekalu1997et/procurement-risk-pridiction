import json

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add matplotlib inline to Cell 1 (index 2)
cell1 = nb['cells'][2]
cell1['source'].insert(4, "%matplotlib inline\n")
cell1['source'].insert(5, "import matplotlib.pyplot as plt\n")

# Update Cell 10 (index 18) - Visualization Pipeline
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
    "print(f\"\\nSHAP plot saved: {shap_plot_path}\")\n",
    "img = Image.open(shap_plot_path)\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.title('SHAP Feature Importance')\n",
    "plt.show()\n",
    "\n",
    "# Generate feature importance plot\n",
    "importance_plot_path = explainability_viz.plot_feature_importance(\n",
    "    explanation.feature_names,\n",
    "    explanation.shap_values,\n",
    "    output_name=\"pipeline_demo_importance\"\n",
    ")\n",
    "print(f\"\\nFeature importance plot saved: {importance_plot_path}\")\n",
    "img = Image.open(importance_plot_path)\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.title('Feature Importance')\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n✓ Visualizations generated\")"
]

# Update Cell 11 (index 20) - Advanced Visualization
cell11 = nb['cells'][20]
cell11['source'] = [
    "print(\"\\n📈 STEP 10: Advanced Visualization\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "from PIL import Image\n",
    "\n",
    "# Create correlation heatmap\n",
    "heatmap_path = visualization_engine.heatmap(\n",
    "    training_df,\n",
    "    output_name=\"pipeline_demo_heatmap\"\n",
    ")\n",
    "print(f\"\\nHeatmap saved: {heatmap_path}\")\n",
    "img = Image.open(heatmap_path)\n",
    "plt.figure(figsize=(12, 8))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.title('Feature Correlation Heatmap')\n",
    "plt.show()\n",
    "\n",
    "# Create histogram\n",
    "hist_path = visualization_engine.histogram(\n",
    "    training_df,\n",
    "    column=\"credit_score\",\n",
    "    output_name=\"pipeline_demo_histogram\"\n",
    ")\n",
    "print(f\"\\nHistogram saved: {hist_path}\")\n",
    "img = Image.open(hist_path)\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.title('Credit Score Distribution')\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n✓ Advanced visualizations generated\")"
]

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Visualization cells updated!")
