import json

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 22 (index 22) - Audit Trail with proper error handling
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
    "# Fetch recent audit events with proper column handling\n",
    "try:\n",
    "    recent_events = auditing.db_connector.fetch_audit_trail(limit=5)\n",
    "    print(\"\\nRecent Audit Events:\")\n",
    "    \n",
    "    # Dynamically check which columns exist\n",
    "    available_cols = ['event_type']\n",
    "    if 'created_at' in recent_events.columns:\n",
    "        available_cols.append('created_at')\n",
    "    elif 'timestamp' in recent_events.columns:\n",
    "        available_cols.append('timestamp')\n",
    "    \n",
    "    print(recent_events[available_cols].head())\n",
    "except Exception as e:\n",
    "    print(f\"\\nNote: Could not fetch audit trail from database: {e}\")\n",
    "    print(\"Audit events are logged to CSV at reports/audit_logs/audit_events_log.csv\")\n",
    "\n",
    "print(\"\\n✓ Pipeline execution logged\")"
]

# Clear execution count and outputs
nb['cells'][22]['execution_count'] = None
nb['cells'][22]['outputs'] = []

with open('notebooks/07_complete_mlops_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Audit trail cell fixed - handles both 'created_at' and 'timestamp' columns")
