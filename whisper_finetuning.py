import os
import argparse
import mlflow
import json
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor
from utils.finetuning_utils import create_dataset, load_model, DataCollatorSpeechSeq2SeqWithPadding
from datasets import load_dataset, concatenate_datasets
#from utils.manifest_utils import apply_preprocessors
import soundfile as sf
import uuid
from peft import LoraConfig, get_peft_model


def main(model_pretrained, train_manifest, val_manifest,data_type):
    # Create necessary directories
    os.makedirs("finetuning/checkpoints", exist_ok=True)
    os.makedirs("finetuning/experiments", exist_ok=True)
    os.makedirs("finetuning/args", exist_ok=True)


    if model_pretrained == 'KBLab/kb-whisper-large':
        config_file = "finetuning/args/whisper_large_args.json"
        with open(config_file, 'r') as f:
            config_json = json.load(f)

        config_json['output_dir'] = "finetuning/checkpoints"
        training_args = Seq2SeqTrainingArguments(# Correctly pass deepspeed config
            **config_json  # Unpack config_json for other arguments like batch_size, etc.
        )
        config_json['output_dir'] = "finetuning/checkpoints"

    elif model_pretrained == 'KBLab/kb-whisper-medium':
        config_file = "finetuning/args/whisper_medium_args.json"
        with open(config_file, 'r') as f:
            config_json = json.load(f)
        config_json['output_dir'] = "finetuning/checkpoints"
        training_args = Seq2SeqTrainingArguments(**config_json)

    elif model_pretrained == 'KBLab/kb-whisper-small':
        config_file = "finetuning/args/whisper_small_args.json"
        with open(config_file, 'r') as f:
            config_json = json.load(f)
        config_json['output_dir'] = "finetuning/checkpoints"
        training_args = Seq2SeqTrainingArguments(**config_json)

    else:
        raise ValueError("Model not supported")

    with open(config_file, 'r') as f:
        config_json = json.load(f)

    checkpoint_name = model_pretrained.split("/")[-1]
    checkpoint_name = f"{checkpoint_name}_{data_type}"
    checkpoint_folder = os.path.join("finetuning/checkpoints", checkpoint_name)
    experiment_folder = os.path.join("finetuning/experiments", checkpoint_name)

    model = load_model(model_pretrained)
    if model_pretrained == 'KBLab/kb-whisper-large':
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)
        
    
    # the Whisper feature extractor performs two operations. 
    # 1. pads/truncates a batch of audio samples such that all samples have an input length of 30s.
    # 2. converting the padded audio arrays to log-Mel spectrograms.
    
    processor = WhisperProcessor.from_pretrained(model_pretrained, language="sv", task="transcribe")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # a manifest is a file that provides metadata about a dataset or model. It often lists the data files and their locations
    #train_dataset = create_dataset(train_manifest)
    #val_dataset = create_dataset(val_manifest)
    
    # def prepare_dataset(batch):
    #     tmp_dir = "/tmp/whisper_audio"
    #     os.makedirs(tmp_dir, exist_ok=True)
    #     audio = batch["audio"]
    #     # Save audio array as a WAV file
    #     file_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.wav")
    #     sf.write(file_path, audio["array"], audio["sampling_rate"])
    #     batch["input_features"] = file_path
    #     # Copy text into a "labels" field for the collator to tokenize later
    #     batch["labels"] = batch["text"]
    #     return batch

    # # Load and preprocess dataset
    # dataset = load_dataset("grdphilip/facebook_m4t_v2_syndata")["train"]
    # dataset = dataset.map(prepare_dataset)

    # # Split into train/validation
    # train_val_split = dataset.train_test_split(test_size=0.2)
    # train_dataset = train_val_split["train"]
    # val_dataset = train_val_split["test"]
    
    train_dataset = create_dataset(train_manifest)
    val_dataset = create_dataset(val_manifest)
    
    print(f"Training model on {len(train_dataset)} samples")
    print(f"Validating model on {len(val_dataset)} samples")
    

    
    # Used for 4 GPUs
    config_json['output_dir'] = checkpoint_folder
    config_json['warmup_steps'] = int(0.5 * len(train_dataset) * 3 / (32 * 4) / 10)
    config_json['save_steps'] = int(len(train_dataset) * 3 / (32 * 4) / 10)
    config_json['eval_steps'] = int(len(train_dataset) * 3 / (32 * 4) / 10)


    print(f"Warming up for {config_json['warmup_steps']} steps")
    
    # the trainer expects a dataset with the following keys: input_features, labels
    # the input_features key should contain the path to the audio file
    # the labels key should contain the text transcription of the audio file
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    
    
    mlflow.set_experiment("finetuning/experiments")
    with mlflow.start_run() as run:
        trainer.train()
        trainer.save_model(checkpoint_folder)
        print(f"Model saved to {checkpoint_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script for Seq2Seq model with Whisper processor")
    parser.add_argument("--model_pretrained", type=str, default="whisper", choices=['KBLab/kb-whisper-large', 'KBLab/kb-whisper-medium', 'KBLab/kb-whisper-small'],
                        help="Pretrained model to fine-tune")
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to the training manifest file")
    parser.add_argument("--val_manifest", type=str, required=True,
                        help="Path to the validation manifest file")
    parser.add_argument("--data_type", type=str, required=True, default="elevenlabs", choices=['elevenlabs-common_voice', 'elevenlabs-fleurs', 'elevenlabs'],)
    args = parser.parse_args()
    
    main(args.model_pretrained, args.train_manifest, args.val_manifest, args.data_type)


"""
First run process_hf_dataset.py to create manifest files
Then run whisper_finetuning.py to start training
python whisper_finetuning.py --model_pretrained KBLab/kb-whisper-small --train_manifest combined_elevenlabs_common_voice_train_manifest.json --val_manifest combined_elevenlabs_common_voice_val_manifest.json --data_type elevenlabs-common_voice
python whisper_finetuning.py --model_pretrained KBLab/kb-whisper-medium --train_manifest combined_elevenlabs_fleurs_train_manifest.json --val_manifest combined_elevenlabs_fleurs_val_manifest.json --data_type elevenlabs-fleurs
python whisper_finetuning.py --model_pretrained KBLab/kb-whisper-large --train_manifest syndata_11labs_train_manifest.json --val_manifest syndata_11labs_val_manifest.json

After training 
Go to terminal and run mlflow ui to see loss and other metrics
"""