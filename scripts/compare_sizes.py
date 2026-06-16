import pandas as pd
import os

models = ['v8n', 'v8s', 'v8m']
results = []

for m in models:
    paths = [
        f'/home/ubuntu/runs/detect/runs/size_comparison/baseball_{m}/results.csv',
        f'/home/ubuntu/runs/size_comparison/baseball_{m}/results.csv'
    ]
    
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            last_row = df.iloc[-1]
            
            results.append({
                'Model': m,
                'mAP@0.5': round(last_row['metrics/mAP50(B)'], 4),
                'mAP@0.5:0.95': round(last_row['metrics/mAP50-95(B)'], 4),
                'Precision': round(last_row['metrics/precision(B)'], 4),
                'Recall': round(last_row['metrics/recall(B)'], 4)
            })
            break

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv('/home/ubuntu/size_comparison_results.csv', index=False)
