import re

import pandas as pd
import requests


def list_models(params):
    base_url = "https://huggingface.co/api/models"

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


# Example usage
params = {
    # "search": "resnet",
    "author": "alzoubi36",
    # "filter": "image-classification",
    "sort": "downloads",
    "direction": "-1",  # descending
    "limit": "200",
    "full": "false",
    "config": "false"
}

models_info = list_models(params)
model_names = [model["modelId"] for model in models_info]
print("Number of current models: ", len(models_info))


# if models_info:
#     for model in models_info:
#         # print(f"Model Name: {model['modelId']}")
#         print(model['modelId'])
#         # print(f"Author: {model['author']}")
#         # print(f"Tags: {', '.join(model['tags'])}")
#         # print(f"Downloads: {model['downloads']}")
#         # print(f"Last Modified: {model['lastModified']}")
#         # print("-" * 40)
# else:
#     print("No models found.")


def get_task_name(model_name):
    names = ['policy_ie_a', 'opp_115', 'piextract', 'policy_detection', 'policy_ie_b', 'policy_qa', 'privacy_qa']
    for name in names:
        match = re.search(rf'pglue_{name}', model_name)
        if match is not None:
            return match.group(0)[6:]
    return ""


def get_model_name(model_name):
    names = ['_priva_t5-v1.1', '_priva_t5', '_t5-v1.1', '_t5']
    for name in names:
        match = re.search(rf'{name}', model_name)
        if match is not None:
            return match.group(0)[1:]
    return ""


def create_tables(model_names):
    sizes = set([name.split('-')[-1] for name in model_names])

    tables = {}
    for size in sizes:
        df = pd.DataFrame(
            index=['policy_ie_a', 'opp_115', 'piextract', 'policy_detection', 'policy_ie_b', 'policy_qa', 'privacy_qa'],
            columns=['t5', 't5-v1.1', 'priva_t5', 'priva_t5-v1.1'])
        df = df.fillna(0)

        for name in model_names:
            name = name.split('/')[-1]
            if size in name:
                task = get_task_name(name)
                model = get_model_name(name)
                if not task:
                    continue

                df.at[task, model] = 1

        tables[size] = df

    return tables


tables = create_tables(model_names)


def pivot_list_to_html_and_save(pivot_dfs, file_path):
    # Convert each pivot DataFrame to an HTML table and store in a list
    html_tables = [pivot_df.to_html(classes='table table-bordered', escape=False) for pivot_df in pivot_dfs]

    # Combine the HTML tables into a single HTML content
    combined_html = "\n".join(html_tables)

    # Save the combined HTML content to a file
    with open(file_path, 'w') as file:
        file.write(combined_html)


def pivot_dict_to_html_and_save(pivot_dict, file_path):
    # Create a CSS style string to improve table appearance
    css_style = """
    <style>
        .table-container {
            display: inline-block;
            margin: 10px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            max-width: 800px; /* Limit table width */
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
    """

    # Convert dictionary items (table name and DataFrame) to HTML with added CSS classes
    html_tables = [
        f"<h2>{table_name}</h2><div class='table-container'>{pivot_df.to_html(classes='table table-bordered', escape=False)}</div><br>"
        for table_name, pivot_df in pivot_dict.items()
    ]

    # Combine the CSS style and HTML tables into a single HTML content
    combined_html = f"{css_style}\n{''.join(html_tables)}"

    # Save the combined HTML content to a file
    with open(file_path, 'w') as file:
        file.write(combined_html)


def pivot_dict_to_markdown_and_save(pivot_dict, file_path):
    # Convert dictionary items (table name and DataFrame) to Markdown content
    markdown_content = f"""## PrivacyGLUE

The finetuning is based on the [PrivacyGLUE](https://github.com/infsys-lab/privacy-glue) dataset proposed by [Shankar et al.](https://www.mdpi.com/2076-3417/13/6/3701).


## Tasks

- OPP-115
- PI-Extract
- Policy-Detection
- PolicyIE-A
- PolicyIE-B
- PolicyQA
- PrivacyQA

## Available Models([here](https://huggingface.co/alzoubi36))\n\n
Number of current models including the 8 pretrained on Privaseer: {len(model_names)}\n\n"""
    for table_name, pivot_df in pivot_dict.items():
        markdown_content += f"### {table_name}\n\n"
        markdown_content += "\n"

        # Replace underscores in index (row names) with tildes (~)
        index_with_tildes = pivot_df.index.to_series().astype(str).str.replace('_', '\_')
        pivot_df.index = index_with_tildes

        markdown_content += pivot_df.to_markdown()
        markdown_content += "\n \n\n"

    # Save the Markdown content to a README.md file
    with open(file_path, 'w') as file:
        file.write(markdown_content)


# Usage example
# pivot_dict_to_html_and_save(tables, f'result.html')

pivot_dict_to_markdown_and_save(tables, f'README.md')
