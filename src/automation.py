import pandas as pd
import re

def inject_metrics_to_readme(report_dict, model_name, readme_path="../README.md"):
    """
    Parses a scikit-learn classification report dictionary into a Markdown table
    and injects it into the README file between designated HTML markers.
    """
    df = pd.DataFrame(report_dict).transpose()
    df = df.round(2)
    
    markdown_table = f"### {model_name} Final Evaluation\n"
    markdown_table += "| Metric | Precision | Recall | F1-Score | Support |\n"
    markdown_table += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for index, row in df.iterrows():
        if index in ['accuracy']:
            continue
        support = int(row['support']) if not pd.isna(row['support']) else ""
        markdown_table += f"| **{index}** | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1-score']:.2f} | {support} |\n"
        
    with open(readme_path, "r") as file:
        readme_content = file.read()
        
    marker_start = f""
    marker_end = f""
    
    pattern = re.compile(rf"{marker_start}.*?{marker_end}", re.DOTALL)
    replacement = f"{marker_start}\n{markdown_table}\n{marker_end}"
    
    updated_readme = re.sub(pattern, replacement, readme_content)
    
    with open(readme_path, "w") as file:
        file.write(updated_readme)