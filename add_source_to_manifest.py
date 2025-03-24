import json

path = "filtering/data/syn11labs_train_manifest.json"
source = "11labs"

filepaths = [
    (path, source)
]

def concat_datasets(filepaths):
    
    # Concatenate datasets
    combined_train_manifest = []
    
    for filepath, source in filepaths:
        with open(filepath) as f:
            data = json.load(f)
            for entry in data:
                entry['source'] = source
            combined_train_manifest.extend(data)
    
    return combined_train_manifest

# Example usage
combined_data = concat_datasets(filepaths)

# Save the combined data to a JSON file
with open(path, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)
