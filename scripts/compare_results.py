import pandas as pd
import os

experiments = ['no_aug', 'basic_aug', 'strong_aug']
results = []

for exp in experiments:
    # 두 경로 모두 확인
    paths = [
        f'/home/ubuntu/runs/detect/runs/comparison/baseball_{exp}/results.csv',
        f'/home/ubuntu/runs/comparison/baseball_{exp}/results.csv'
    ]
    
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            last_row = df.iloc[-1]
            
            results.append({
                'Experiment': exp,
                'mAP@0.5': round(last_row['metrics/mAP50(B)'], 4),
                'mAP@0.5:0.95': round(last_row['metrics/mAP50-95(B)'], 4),
                'Precision': round(last_row['metrics/precision(B)'], 4),
                'Recall': round(last_row['metrics/recall(B)'], 4)
            })
            break

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv('/home/ubuntu/comparison_results.csv', index=False)
