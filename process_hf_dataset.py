import argparse
from utils.text_preprocessing_utils import apply_preprocessors
from utils.manifest_utils import convert_hf_dataset_to_manifest, read_manifest_file, write_manifest_file, remove_special_samples, convert_finetuning_manifest_to_filtering_manifest
from datasets import load_dataset
import os

def main(args):

    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/wav_data", exist_ok=True)
    os.makedirs("data/manifest_data", exist_ok=True)
    os.makedirs("data/manifest_data/finetuning/raw", exist_ok=True)
    os.makedirs("data/manifest_data/finetuning/preprocessed", exist_ok=True)

    # Load environment variables
    # Load dataset
    cv_dataset = load_dataset(args.dataset_name, args.dataset_language, split=args.split, token=args.token)
    # Will contain the columns audio, text, duration
    
    
    # Convert dataset to manifest file
    convert_hf_dataset_to_manifest(dataset=cv_dataset, dataset_type=args.dataset_type, manifest_filename=args.manifest_filename)
    print("Converted hugginface dataset to manifest file")
    
    # Read the generated manifest file
    train_manifest = read_manifest_file(args.manifest_filename)
    print("Loaded the generated manifest file")
    
    # Apply preprocessors to the manifest data
    train_manifest_processed = apply_preprocessors(train_manifest)
    print("Preprocessed dataset")
    
    # Remove special samples from the processed manifest
    train_manifest_processed = remove_special_samples(train_manifest_processed)
    print("Removed special samples from dataset")
    
    # Write the processed manifest back to a file
    write_manifest_file(train_manifest_processed, args.manifest_filename)
    print("Wrote processed manifest to file")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and create a processed manifest file.")
    
    # Required arguments
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to load")
    parser.add_argument("dataset_language", type=str, help="Language of the dataset")
    parser.add_argument("split", type=str, help="Split of the dataset to use (e.g., train, test)")
    parser.add_argument("dataset_type", type=str, help="Type of dataset (e.g., common_voice)")
    parser.add_argument("manifest_filename", type=str, help="Filename for the generated manifest file")
    parser.add_argument("token", type=str, help="Token for the dataset")
    
    args = parser.parse_args()
    
    main(args)

    #python process_hf_dataset.py grdphilip/elevenlabs_syndata default customized syndata_11labs_train_manifest.json 
    #python process_hf_dataset.py google/fleurs sv-se test fleurs fleurs_val_manifest.json 
    #python process_hf_dataset.py grdphilip/northvolt_syndata default train customized northvolt_train_manifest.json 
