import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
from utils.evaluation_utils import create_dataloaders, DataCollatorSpeechSeq2SeqWithPadding, calculate_and_store_metrics, clean_entities
from tqdm import tqdm
import pandas as pd

def main(args):

    pretrained_model = args.pretrained_model
    base_model = args.base_model
    save_name = args.save_name
    batch_size = args.batch_size
    checkpoint_path = "/home/ec2-user/SageMaker/swedish_semantic_audio_filtering/finetuning/checkpoints/checkpoint-471/"

    # Torch configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model from checkpoint
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint_path,  # Instead of `pretrained_model`, use the checkpoint directory
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model)

    # Define the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        generate_kwargs={"task": "transcribe", "language": "swedish"}
    )

    # Define data files and subsets (you can customize these paths as per your setup)
    # filenames = [
    #     'data/mls_manifest_processed.json',
    #     'data/fleurs_manifest_processed.json',
    #     'data/bracarense_manifest_processed.json',
    #     'data/common_voice_manifest_processed.json',
    #     'data/val_manifest_wps_processed.json'
    # ]

    # prints = ["MLS", "FLEURS", "BRACARENSE", "CV", "VALIDATION"]
    
   
    filenames = ["data/manifest_data/finetuning/raw/fleurs_swedish_with_entities_eval_manifest.json"]
    #filenames = ["data/manifest_data/finetuning/raw/cv_swedish_with_entities_eval_manifest.json"]
    
    prints = ["TEST"]

    # Create dataloaders and corresponding dataframes
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    dataloaders, dataframes = create_dataloaders(filenames, batch_size, data_collator)

    # Define JWER transformations
    normalize_transforms = Compose([
        RemoveEmptyStrings(),
        ToLowerCase(),
        RemoveMultipleSpaces(),
        Strip(),
        RemovePunctuation(),
        ReduceToListOfListOfWords(),
    ])

    not_normalize_transforms = Compose([
        RemoveEmptyStrings(),
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToListOfListOfWords(),
    ])
    
    

    # Initialize DataFrames for results
    normalized_results_df = pd.DataFrame(columns=["SUBSET", "WER", "CER", "ENTITY_ACCURACY"])
    not_normalized_results_df = pd.DataFrame(columns=["SUBSET", "WER", "CER", "ENTITY_ACCURACY"])

    # Process each subset
    for dataloader, dataframe, subset_name in zip(dataloaders, dataframes, prints):
        # Collect candidates from the dataloader
        candidates = []
        for batch in tqdm(dataloader, desc=f"Processing {subset_name} audio files"):
            candidates.extend(pipe(batch))

        # Get reference texts
        references = dataframe['text'].to_list()
        reference_entities = dataframe['entities'].to_list()
        metadata = dataframe['metadata'].to_list()
        
        # Clean entities
        print(reference_entities)
        #entities = clean_entities(reference_entities)
        print(references)
        #print(entities)

        # Calculate and store normalized metrics
        normalized_results_df, missed_entities_df_norm = calculate_and_store_metrics(references, candidates, reference_entities, metadata, normalize_transforms, subset_name, normalized_results_df, normalized=True)

        # Calculate and store non-normalized metrics
        not_normalized_results_df, missed_entities_df_not_norm = calculate_and_store_metrics(references, candidates, reference_entities, metadata, not_normalize_transforms, subset_name, not_normalized_results_df, normalized=False)

    # Save results to CSV files
    normalized_results_df.to_csv(f"results/normalized_results_{save_name}.csv", index=False)
    not_normalized_results_df.to_csv(f"results/not_normalized_results_{save_name}.csv", index=False)
    missed_entities_df_norm.to_csv(f"results/missed_entities_norm_{save_name}.csv", index=False)
    missed_entities_df_not_norm.to_csv(f"results/missed_entities_not_norm_{save_name}.csv", index=False)

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech recognition model evaluation script")
    parser.add_argument("pretrained_model", type=str, help="Path to the pretrained model")
    parser.add_argument("base_model", type=str, help="Path to the base model")
    parser.add_argument("save_name", type=str, help="Name to save the results")
    parser.add_argument("batch_size", type=int, help="Batch size for processing")
    
    args = parser.parse_args()
    main(args)

# python whisper_evaluation.py KBLab/kb-whisper-small KBLab/kb-whisper-small finetuned_benchmark_cv_small 32

#normalized
# entity_score 0.9597701149425287
# wer_score 0.051224944320712694
# cer_score 0.015815370196813495

#not normalized
# entity_score 0.9540229885057471
# wer_score 0.19481429572529782
# cer_score 0.03455004591368228

# python whisper_evaluation.py KBLab/kb-whisper-medium KBLab/kb-whisper-medium syndata_11labs_medium 4

#normalized
# entity_score 0.9655172413793104
# wer_score 0.0400890868596882
# cer_score 0.012769447047797563

#not normalized
# entity_score 0.9425287356321839
# wer_score 0.1920112123335669
# cer_score 0.03213957759412305

# python whisper_evaluation.py KBLab/kb-whisper-large KBLab/kb-whisper-large syndata_11labs_medium 4

#normalized
# entity_score 0.9712643678160919
# wer_score 0.042316258351893093
# cer_score 0.009840674789128397

#not normalized

# entity_score 0.9655172413793104
# wer_score 0.18990889978976874
# cer_score 0.02835169880624426
