import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
from utils.evaluation_utils import create_dataloaders, DataCollatorSpeechSeq2SeqWithPadding, calculate_and_store_metrics, clean_entities
from tqdm import tqdm
import pandas as pd
import os

def main(args):

    pretrained_model = args.pretrained_model
    base_model = args.base_model
    save_name = args.save_name
    batch_size = args.batch_size
    if os.path.exists(f"/home/ec2-user/SageMaker/swedish_semantic_audio_filtering/finetuning/checkpoints/{pretrained_model}"):
        model_path = f"/home/ec2-user/SageMaker/swedish_semantic_audio_filtering/finetuning/checkpoints/{pretrained_model}"
    else:
        model_path = pretrained_model

    # Torch configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model from checkpoint
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,  # Instead of `pretrained_model`, use the checkpoint directory
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
    
   
    #filenames = ["data/manifest_data/finetuning/raw/fleurs_swedish_with_entities_eval_manifest.json"]
    #filenames = ["data/manifest_data/finetuning/raw/cv_swedish_with_entities_eval_manifest.json"]
    #filenames = ["data/manifest_data/finetuning/raw/common_voice_test_manifest.json"]
    #filenames = ["data/manifest_data/finetuning/raw/fleurs_val_manifest.json"]
    filenames = ["data/manifest_data/finetuning/raw/northvolt_train_manifest.json"]
    
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
    all_candidates_references_df = pd.DataFrame(columns=["SUBSET", "REFERENCE", "CANDIDATE", "REFERENCE_ENTITIES"])

    # Process each subset
    for dataloader, dataframe, subset_name in zip(dataloaders, dataframes, prints):
        # Collect candidates from the dataloader
        candidates = []
        for batch in tqdm(dataloader, desc=f"Processing {subset_name} audio files"):
            candidates.extend(pipe(batch))

        # Get reference texts
        references = dataframe['text'].to_list()
        
        has_entities = 'entities' in dataframe.columns and 'metadata' in dataframe.columns
        if has_entities:
            reference_entities = dataframe['entities'].to_list()
            metadata = dataframe['metadata'].to_list()
        else:
            reference_entities = None
            metadata = None

        # Store all candidates, references, and reference entities
        subset_df = pd.DataFrame({
            "SUBSET": [subset_name] * len(references),
            "REFERENCE": references,
            "CANDIDATE": candidates,
            "REFERENCE_ENTITIES": reference_entities if has_entities else [None] * len(references)
        })
        all_candidates_references_df = pd.concat([all_candidates_references_df, subset_df], ignore_index=True)

        # Calculate and store normalized metrics
        normalized_results_df, missed_entities_df_norm = calculate_and_store_metrics(references, candidates, reference_entities, metadata, normalize_transforms, subset_name, normalized_results_df, normalized=True)

        # Calculate and store non-normalized metrics
        not_normalized_results_df, missed_entities_df_not_norm = calculate_and_store_metrics(references, candidates, reference_entities, metadata, not_normalize_transforms, subset_name, not_normalized_results_df, normalized=False)

    # Save results to CSV files
    normalized_results_df.to_csv(f"benchmark_results/normalized_results_{save_name}.csv", index=False)
    not_normalized_results_df.to_csv(f"benchmark_results/not_normalized_results_{save_name}.csv", index=False)
    all_candidates_references_df.to_csv(f"benchmark_results/all_candidates_references_{save_name}.csv", index=False)
    
    if 'entities' in dataframe.columns:
        missed_entities_df_norm.to_csv(f"benchmark_results/missed_entities_norm_{save_name}.csv", index=False)
        missed_entities_df_not_norm.to_csv(f"benchmark_results/missed_entities_not_norm_{save_name}.csv", index=False)
        
    print(f"Results saved to CSV {save_name}")
    
    
if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech recognition model evaluation script")
    parser.add_argument("pretrained_model", type=str, help="Path to the pretrained model")
    parser.add_argument("base_model", type=str, help="Path to the base model")
    parser.add_argument("save_name", type=str, help="Name to save the results")
    parser.add_argument("batch_size", type=int, help="Batch size for processing")
    
    args = parser.parse_args()
    main(args)
    
    
# =============================================================================================================
"""
Run with openai whisper model as base model on the commonvoice dataset for eval
python whisper_evaluation.py openai/whisper-large-v3 openai/whisper-large-v3 tonar_res_cv_large 32

CV! 
#normalized
entity_score 0.758 (0.05 Sämre) 
wer: 0.112 (0.05 Sämre) 

#not normalized
entity_score 0.708 (Sämre)
wer: 0.157 (0.053 sämre)

FLEURS!
#normalized
entity_score 0.710 (Bättre)
wer: 0.0941 (Bättre)

#not_normalized
entity_score 0.6725 (Bättre)
wer: 0.1578 (Sämre)

FINE TUNED
#normalized
entity_score 0.6981327800829875
wer_score 0.18090783319059225

#not normalied
entity_score 0.673582295988935
wer_score 0.23778201670722884


python whisper_evaluation.py openai/whisper-large-v3 openai/whisper-large-v3 northvolt_res_cv_large 32
"""

# =============================================================================================================
# python whisper_evaluation.py KBLab/kb-whisper-small KBLab/kb-whisper-small finetuned_benchmark__small 32

#normalized
# entity_score 0.9597701149425287
# wer_score 0.051224944320712694
# cer_score 0.015815370196813495

#not normalized
# entity_score 0.9540229885057471
# wer_score 0.19481429572529782
# cer_score 0.03455004591368228

# python whisper_evaluation.py kb-whisper-small_elevenlabs-common_voice KBLab/kb-whisper-small finetuned_syncv_fleurs_small 32

#normalized
# entity_score 0.9655172413793104
# wer_score 0.0400890868596882
# cer_score 0.012769447047797563

#not normalized
# entity_score 0.9425287356321839
# wer_score 0.1920112123335669
# cer_score 0.03213957759412305

# python whisper_evaluation.py kb-whisper-large_elevenlabs KBLab/kb-whisper-large save_name 4

#normalized
# entity_score 0.9712643678160919
# wer_score 0.042316258351893093
# cer_score 0.009840674789128397

#not normalized

# entity_score 0.9655172413793104
# wer_score 0.18990889978976874
# cer_score 0.02835169880624426
