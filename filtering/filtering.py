import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor
from tqdm import tqdm
import pickle
from model import MusCALL
from dataset import AudioCaptionDataset
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def compute_cosine_similarity(audio_features, text_features):
    cosine_sim = F.cosine_similarity(text_features, audio_features, dim=1)  # No loop needed
    cosine_sim_bounded = (1 + cosine_sim) / 2  # Bound to [0, 1]
    return cosine_sim_bounded

def load_embeddings():
    # Check if the embeddings files exist
    if not os.path.exists("embeddings/synthetic_audio_features.pkl") or not os.path.exists("embeddings/synthetic_text_features.pkl"):
        print("Embeddings files not found. Please run the extract_embeddings() method first.")
        return

    audio_features_path = "embeddings/" + "synthetic_audio_features.pkl"
    text_features_path = "embeddings/" + "synthetic_text_features.pkl"

    with open(audio_features_path, 'rb') as af_file:
        audio_features = pickle.load(af_file)

    with open(text_features_path, 'rb') as tf_file:
        text_features = pickle.load(tf_file)

    return audio_features, text_features
    

class FilteringFramework:
    def __init__(self, config, pretrained_model_path):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.training.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("KBLab/kb-whisper-medium")
        self.checkpoint_path = pretrained_model_path
        

        self.path_to_model = os.path.join(
            self.config.env.experiments_dir,
            self.config.env.experiment_id,
            "best_model.pth.tar",
        )
        
        print(f"Loading model from: {self.path_to_model}")
                
        self.set_seed()
        self.load_dataset()
        self.load_model()
        self.similarities = None

    def collate_fn(self, batch):
        input_audio, text_input_ids, text_attention_mask, idx, sources = zip(*batch)
        
        original_mel_spectograms = self.feature_extractor(input_audio, sampling_rate=16000, max_length=480000, return_tensors="pt").input_features

        text_input_ids = torch.stack(text_input_ids)
        text_attention_mask = torch.stack(text_attention_mask)

        max_len = max([len(i) for i in input_audio])

        original_audio = []
        for audio in input_audio:
            if len(audio) < max_len:
                zeros_needed = np.zeros(max_len - len(audio))
                audio = np.concatenate((audio, zeros_needed), axis=0)
                original_audio.append(audio)
            else:    
                original_audio.append(audio)

        original_audio = np.stack(original_audio)

        return {"input_audio": original_mel_spectograms.to(self.device), \
                "original_audio": original_audio, \
                "text_input_ids": text_input_ids.to(self.device), \
                "text_attention_mask": text_attention_mask.to(self.device),\
                "idx": idx,\
                "sources": sources}
    
    def load_dataset(self):
        dataset = AudioCaptionDataset(self.config.dataset_config, dataset_type="to_filter")
        self.batch_size = 32
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def load_model(self):
        self.model = MusCALL(self.config.model_config)
        print(f"Loading model from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        print(checkpoint["epoch"])
        print(checkpoint.keys())
        print(checkpoint["epoch"])
        self.model.load_state_dict(checkpoint, strict=False)
        print(self.model)
        self.model.to(self.device)
        self.model.eval()
        
        

    def set_seed(self,seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        


    def extract_embeddings(self):
        # Create a directory to save the features called "embeddings"
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        audio_features_path = "embeddings/" + "synthetic_audio_features.pkl"

        text_features_path = "embeddings/" + "synthetic_text_features.pkl"
        
        dataset_size = len(self.data_loader.dataset)

        all_audio_features = torch.zeros(dataset_size, 512, device=self.device)
        all_text_features = torch.zeros(dataset_size, 512, device=self.device)


        total_samples_processed = 0

        print(self.model)
        
        for batch in tqdm(self.data_loader, desc="Loading data", leave=False):
            original_mel_spectograms = batch["input_audio"].to(self.device)
            text_input_ids = batch["text_input_ids"].to(self.device)
            text_attention_mask = batch["text_attention_mask"].to(self.device)
            

            with torch.no_grad():
                audio_features = self.model.encode_audio(original_mel_spectograms)
                text_features = self.model.encode_text(text_input_ids, text_attention_mask)

            batch_size = audio_features.size(0)
            
            all_audio_features[total_samples_processed:total_samples_processed + batch_size] = audio_features
            all_text_features[total_samples_processed:total_samples_processed + batch_size] = text_features
            

            total_samples_processed += batch_size
            

        # Convert tensors to CPU before saving to avoid GPU-related issues in the pickle files
        audio_features = all_audio_features.cpu()
        text_features = all_text_features.cpu()
    

        print(f"Total samples processed: {total_samples_processed}")
        print(f"First few audio embeddings: {all_audio_features[:5, :5]}")
        print(f"First few text embeddings: {all_text_features[:5, :5]}")
        

        # Save the features to pickle files
        with open(audio_features_path, 'wb') as af_file:
            pickle.dump(audio_features, af_file)

        with open(text_features_path, 'wb') as tf_file:
            pickle.dump(text_features, tf_file)

        return audio_features, text_features


    def apply_filtering(self, synthetic_data_manifest, stdev_threshold):
        
        mean = np.mean(self.similarities)
        stdev = np.std(self.similarities)

        # Identify the condition for outliers
        condition = self.similarities < (mean - stdev_threshold * stdev)

        # Get the indices of the outliers
        outlier_indices = np.where(condition)[0]

        # Get the samples to delete
        samples_to_delete = []
        audio_durations = 0
        for i, sample in enumerate(synthetic_data_manifest):
            if i in outlier_indices:
                samples_to_delete.append({"id": sample['audio_id'], "path": sample['audio_path']})    
                audio_path = sample['audio_path']
                audio, sr = librosa.load(audio_path, sr=None)  # Load the audio file
                duration = librosa.get_duration(y=audio, sr=sr)  # Get the duration in seconds
                audio_durations += duration

        # Convert to minutes
        audio_durations = audio_durations / 60

        print(f"Total number of samples to delete: {len(samples_to_delete)}")
        print(f"Total audio duration to delete: {audio_durations} minutes")

        return samples_to_delete
    
        
    def get_similarities(self, audio_features, text_features):        
        similarities = []
        for embedding_audio, embedding_text in zip(audio_features, text_features):
            similarities.append(compute_cosine_similarity(embedding_text.unsqueeze(0), embedding_audio.unsqueeze(0)))
        
        similarities_tensor = torch.tensor(similarities)
        self.similarities = similarities_tensor.numpy()
        


    def run(self, data_manifest_path, stdev_threshold=3):
    # Extract embeddings
        audio_features, text_features = self.extract_embeddings()

        # Compute similarities
        self.get_similarities(audio_features, text_features)
        samples_to_delete = self.apply_filtering(data_manifest_path, stdev_threshold)
        print(f"Total number of samples to delete: {len(samples_to_delete)}")

        
        save_dir = os.path.dirname(data_manifest_path)
        save_path = os.path.join(save_dir, "dist_fb.png")
        
        print(f"Max similarity: {max(self.similarities)}")
        print(f"Min similarity: {min(self.similarities)}")
        print(f"Mean similarity: {np.mean(self.similarities)}")
        
        sources=["11labs"] * len(self.similarities)

        #Unique sources and their colors
        unique_sources = list(set(sources))
        colors = plt.cm.get_cmap("viridis", len(unique_sources))  # Use tab10 for distinct colors
        source_colors = {src: colors(i) for i, src in enumerate(unique_sources)}
        print(len(self.similarities))
        # Plot the distribution of similarity values
        plt.figure(figsize=(12, 8))
        for src in unique_sources:
            # Get similarity values for this source
            source_similarities = [self.similarities[i] for i in range(len(sources)) if sources[i] == src]
            
            
            plt.hist(
                source_similarities, 
                bins=30, 
                color=source_colors[src], 
                edgecolor="black", 
                alpha=0.7, 
                label=src  # Label each source
            )

        plt.title("Distribution of Similarity Values", fontsize=16)
        plt.xlabel("Similarity", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title="Source")  # Add legend for source categories
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        audio_features = audio_features.detach().cpu().numpy()
        text_features = text_features.detach().cpu().numpy()

        n_samples = min(len(audio_features), len(text_features))
        perplexity_value = min(30, n_samples - 1)  # Ensure valid perplexity

        if n_samples < 2:
            print("Not enough samples for t-SNE visualization.")
            return

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
        audio_2d = tsne.fit_transform(audio_features)
        text_2d = tsne.fit_transform(text_features)

        # Plot t-SNE results
        plt.figure(figsize=(8, 6))
        for src in unique_sources:
            # Get indices for this source
            idx = [j for j, s in enumerate(sources) if s == src]
            
            # Plot text embeddings (X markers)
            plt.scatter(
                text_2d[idx, 0], text_2d[idx, 1], 
                color=source_colors[src], 
                marker='x', 
                label=f"Text - {src}", 
                alpha=0.6, s=20
            )

            # Plot audio embeddings (O markers)
            plt.scatter(
                audio_2d[idx, 0], audio_2d[idx, 1], 
                color=source_colors[src], 
                marker='o', 
                label=f"Audio - {src}", 
                edgecolors='black', 
                alpha=0.6, s=20
            )

        plt.legend()
        plt.title("t-SNE Visualization of Text & Audio Embeddings")

        # Save to the same directory as `data_manifest_path`
        save_path = os.path.join(save_dir, "tsne_plot.png")
        plt.savefig(save_path)
        print(f"t-SNE plot saved to: {save_path}")

        plt.close()
