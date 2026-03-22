import pandas as pd
import re
import json

def inject_metrics_to_readme(report_dict, model_name, readme_path="README.md"): 
    df = pd.DataFrame(report_dict).transpose()
    df = df.round(2)

    markdown_table = f"### {model_name} Final Evaluation\n"
    markdown_table += "| Metric | Precision | Recall | F1-Score | Support |\n"  
    markdown_table += "| :--- | :--- | :--- | :--- | :--- |\n"

    for index, row in df.iterrows():
        if index in ['accuracy']: continue
        support = int(row['support']) if not pd.isna(row['support']) else ""    
        markdown_table += f"| **{index}** | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1-score']:.2f} | {support} |\n"

    with open(readme_path, "r") as file:
        readme_content = file.read()

    marker_start = f"<!-- [[{model_name}_START]] -->"
    marker_end = f"<!-- [[{model_name}_END]] -->"

    escaped_start = re.escape(marker_start)
    escaped_end = re.escape(marker_end)

    pattern = re.compile(rf"{escaped_start}.*?{escaped_end}", re.DOTALL)        
    replacement = f"{marker_start}\n\n{markdown_table}\n{marker_end}"

    if re.search(pattern, readme_content):
        updated_readme = re.sub(pattern, replacement, readme_content)
        with open(readme_path, "w") as file:
            file.write(updated_readme)

def inject_leaderboard_to_readme(top_configs, model_name, readme_path="README.md"):
    if not top_configs: return

    markdown_table = f"### {model_name} Top 5 Optimization Leaderboard\n"       
    
    param_keys = list(top_configs[0]['params'].keys())
    if 'metrics' in top_configs[0]:
        metric_keys = list(top_configs[0]['metrics'].keys())
    else:
        metric_keys = [top_configs[0].get('metric_name', 'Metric')]

    markdown_table += "| Rank | " + " | ".join(param_keys) + " | " + " | ".join(metric_keys) + " |\n"
    markdown_table += "| :--- " * (len(param_keys) + len(metric_keys) + 1) + "|\n"

    for i, conf in enumerate(top_configs):
        row_vals = [str(i+1)]
        for k in param_keys:
            val = conf['params'][k]
            if val is None: row_vals.append('None')
            elif isinstance(val, float): row_vals.append(f"{val:.6f}")
            else: row_vals.append(str(val))
                
        if 'metrics' in conf:
            for mk in metric_keys:
                m_val = conf['metrics'][mk]
                row_vals.append(f"{m_val:.6f}" if isinstance(m_val, float) else str(m_val))
        else:
            row_vals.append(f"{conf.get('metric', 0):.6f}")

        markdown_table += "| " + " | ".join(row_vals) + " |\n"

    with open(readme_path, "r") as file:
        readme_content = file.read()

    marker_start = f"<!-- [[{model_name}_LEADERBOARD_START]] -->"
    marker_end = f"<!-- [[{model_name}_LEADERBOARD_END]] -->"

    escaped_start = re.escape(marker_start)
    escaped_end = re.escape(marker_end)

    pattern = re.compile(rf"{escaped_start}.*?{escaped_end}", re.DOTALL)        
    replacement = f"{marker_start}\n\n{markdown_table}\n{marker_end}"

    if re.search(pattern, readme_content):
        updated_readme = re.sub(pattern, replacement, readme_content)
        with open(readme_path, "w") as file:
            file.write(updated_readme)

def update_readme_from_json(json_path="results/best_configurations.json"):      
    try:
        with open(json_path, "r") as f: data = json.load(f)
    except Exception as e:
        print(f"Could not load {json_path}: {e}")
        return

    print("Parsing JSON and injecting tabular data into README.md...")

    if "TREE_ENSEMBLE" in data and "metrics" in data["TREE_ENSEMBLE"]:
        inject_metrics_to_readme(data["TREE_ENSEMBLE"]["metrics"], "TREE")      
    if "ALERTING_LSTM" in data and "metrics" in data["ALERTING_LSTM"]:
        inject_metrics_to_readme(data["ALERTING_LSTM"]["metrics"], "LSTM")      
    if "PREDICTIVE_LSTM" in data and "metrics" in data["PREDICTIVE_LSTM"]:      
        inject_metrics_to_readme(data["PREDICTIVE_LSTM"]["metrics"], "UNSUPERVISED_LSTM")

    if "TREE_ENSEMBLE" in data and "top_5_configs" in data["TREE_ENSEMBLE"] and data["TREE_ENSEMBLE"]["top_5_configs"]:
        inject_leaderboard_to_readme(data["TREE_ENSEMBLE"]["top_5_configs"], "TREE")
    if "ALERTING_LSTM" in data and "top_5_configs" in data["ALERTING_LSTM"] and data["ALERTING_LSTM"]["top_5_configs"]:
        inject_leaderboard_to_readme(data["ALERTING_LSTM"]["top_5_configs"], "ALERTING_LSTM")
    if "PREDICTIVE_LSTM" in data and "top_5_configs" in data["PREDICTIVE_LSTM"] and data["PREDICTIVE_LSTM"]["top_5_configs"]:
        inject_leaderboard_to_readme(data["PREDICTIVE_LSTM"]["top_5_configs"], "PREDICTIVE_LSTM")

    print("README.md successfully updated!")

if __name__ == "__main__":
    update_readme_from_json()
