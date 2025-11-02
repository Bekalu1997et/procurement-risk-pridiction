import json

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 16 (index 16) - Recommendation Pipeline (Step 8)
nb['cells'][16]['source'] = [
    "print(\"\\n STEP 8: Recommendation Pipeline\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "# Generate recommendations\n",
    "recommendations = recommendation.build_recommendations(\n",
    "    risk_level=explanation.risk_level,\n",
    "    top_features=explanation.top_features\n",
    ")\n",
    "\n",
    "print(\"\\nActionable Recommendations:\")\n",
    "for i, reco in enumerate(recommendations, 1):\n",
    "    print(f\"  {i}. {reco}\")\n",
    "\n",
    "print(\"\\n✓ Recommendations generated\")"
]

# Cell 18 (index 18) - Visualization Pipeline (Step 9)
nb['cells'][18]['source'] = [
    "print(\"\\n📊 STEP 9: Visualization Pipeline\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Create subplots for feature importance\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 6))\n",
    "\n",
    "# Left: SHAP bar plot\n",
    "top_features = sorted(zip(explanation.feature_names, abs(explanation.shap_values)), key=lambda x: x[1], reverse=True)[:10]\n",
    "features, values = zip(*top_features)\n",
    "axes[0].barh(range(len(features)), values, color='steelblue')\n",
    "axes[0].set_yticks(range(len(features)))\n",
    "axes[0].set_yticklabels([f.replace('numeric__', '').replace('categorical__', '') for f in features])\n",
    "axes[0].invert_yaxis()\n",
    "axes[0].set_xlabel('|SHAP Value|', fontsize=12)\n",
    "axes[0].set_title('Top 10 SHAP Feature Importance', fontsize=14, fontweight='bold')\n",
    "axes[0].grid(axis='x', alpha=0.3)\n",
    "\n",
    "# Right: Feature importance with direction\n",
    "top_features_signed = sorted(zip(explanation.feature_names, explanation.shap_values), key=lambda x: abs(x[1]), reverse=True)[:10]\n",
    "features_s, values_s = zip(*top_features_signed)\n",
    "colors = ['red' if v > 0 else 'green' for v in values_s]\n",
    "axes[1].barh(range(len(features_s)), values_s, color=colors, alpha=0.7)\n",
    "axes[1].set_yticks(range(len(features_s)))\n",
    "axes[1].set_yticklabels([f.replace('numeric__', '').replace('categorical__', '') for f in features_s])\n",
    "axes[1].invert_yaxis()\n",
    "axes[1].set_xlabel('SHAP Value (Red=Increase Risk, Green=Decrease Risk)', fontsize=12)\n",
    "axes[1].set_title('Feature Impact Direction', fontsize=14, fontweight='bold')\n",
    "axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)\n",
    "axes[1].grid(axis='x', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n✓ Feature importance visualizations displayed\")"
]

# Cell 20 (index 20) - Advanced Visualization (Step 10)
nb['cells'][20]['source'] = [
    "print(\"\\n📈 STEP 10: Advanced Visualization\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Pairplot - select key numeric features\n",
    "plot_cols = ['credit_score', 'annual_spend', 'late_ratio', 'dispute_rate', 'risk_label']\n",
    "plot_df = training_df[plot_cols].sample(n=min(500, len(training_df)), random_state=42)\n",
    "\n",
    "print(\"\\nGenerating pairplot (this may take a moment)...\")\n",
    "g = sns.pairplot(plot_df, hue='risk_label', palette={'low': 'green', 'medium': 'orange', 'high': 'red'}, \n",
    "                 diag_kind='kde', plot_kws={'alpha': 0.6}, height=2.5)\n",
    "g.fig.suptitle('Feature Relationships by Risk Level', y=1.02, fontsize=16, fontweight='bold')\n",
    "plt.show()\n",
    "\n",
    "# Subplots: Heatmap and Histogram\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 6))\n",
    "\n",
    "# Left: Correlation heatmap\n",
    "numeric_cols = training_df.select_dtypes(include=['float64', 'int64']).columns[:10]\n",
    "corr = training_df[numeric_cols].corr()\n",
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0], \n",
    "            cbar_kws={'label': 'Correlation'})\n",
    "axes[0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')\n",
    "\n",
    "# Right: Credit score distribution by risk\n",
    "for risk in ['low', 'medium', 'high']:\n",
    "    data = training_df[training_df['risk_label'] == risk]['credit_score']\n",
    "    axes[1].hist(data, bins=30, alpha=0.6, label=risk.capitalize())\n",
    "axes[1].set_xlabel('Credit Score', fontsize=12)\n",
    "axes[1].set_ylabel('Frequency', fontsize=12)\n",
    "axes[1].set_title('Credit Score Distribution by Risk Level', fontsize=14, fontweight='bold')\n",
    "axes[1].legend()\n",
    "axes[1].grid(axis='y', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n✓ Advanced visualizations displayed\")"
]

# Cell 22 (index 22) - Audit Trail (Step 11) - Fixed
nb['cells'][22]['source'] = [
    "print(\"\\n📝 STEP 11: Audit Trail\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "# Log complete pipeline execution\n",
    "auditing.persist_audit_log(\n",
    "    event_type=\"complete_pipeline_execution\",\n",
    "    payload={\n",
    "        \"supplier_id\": \"DEMO_001\",\n",
    "        \"prediction\": result['prediction'],\n",
    "        \"confidence\": explanation.confidence,\n",
    "        \"top_features\": [f[0] for f in explanation.top_features[:3]],\n",
    "        \"recommendations_count\": len(recommendations),\n",
    "        \"visualizations_generated\": 4\n",
    "    }\n",
    ")\n",
    "\n",
    "# Fetch recent audit events\n",
    "try:\n",
    "    recent_events = auditing.db_connector.fetch_audit_trail(limit=5)\n",
    "    print(\"\\nRecent Audit Events:\")\n",
    "    # Check if timestamp or created_at column exists\n",
    "    time_col = 'created_at' if 'created_at' in recent_events.columns else 'timestamp'\n",
    "    if time_col in recent_events.columns:\n",
    "        print(recent_events[['event_type', time_col]].head())\n",
    "    else:\n",
    "        print(recent_events[['event_type']].head())\n",
    "except Exception as e:\n",
    "    print(f\"\\nNote: Could not fetch audit trail from database: {e}\")\n",
    "    print(\"Audit events are still logged to CSV at reports/audit_logs/\")\n",
    "\n",
    "print(\"\\n✓ Pipeline execution logged\")"
]

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cells fixed:")
print("  - Cell 9 (Step 8): Recommendations")
print("  - Cell 10 (Step 9): Feature importance with subplots")
print("  - Cell 11 (Step 10): Pairplot + heatmap/histogram")
print("  - Cell 12 (Step 11): Audit trail with timestamp fix")
