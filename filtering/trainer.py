import os
import time
import numpy as np
from tqdm import tqdm
import torch
import mlflow
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sentence_transformers import SentenceTransformer

from base_trainer import BaseTrainer
from model import MusCALL
from dataset import AudioCaptionDataset
from transformers import WhisperFeatureExtractor


class MusCALLTrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.batch_size = self.config.training.dataloader.batch_size

        self.load()

        self.global_step = 0

        self.scaler = torch.cuda.amp.GradScaler()

        # https://huggingface.co/KBLab/sentence-bert-swedish-cased blir det problem med cased?
        # https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/
        # https://huggingface.co/intfloat/multilingual-e5-large-instruct
        
        self.sbert_model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
        self.sbert_model.to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("KBLab/kb-whisper-medium")


    def collate_fn(self, batch):
        #_, input_audio, text_input_ids, original_audio, _, text_attention_mask, idx = zip(*batch)  
        input_audio, text_input_ids, text_attention_mask, idx = zip(*batch)
        
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
                "idx": idx}
    
    
    def load_dataset(self):
        self.logger.write("Loading dataset")
        dataset_name = self.config.dataset_config.dataset_name
        

        if dataset_name == "common_voice":
            self.train_dataset = AudioCaptionDataset(self.config.dataset_config)
            self.val_dataset = AudioCaptionDataset(self.config.dataset_config, dataset_type="test")
        else:
            raise ValueError("{} dataset is not supported.".format(dataset_name))
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            **self.config.training.dataloader,
            drop_last=True,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            **self.config.training.dataloader,
            drop_last=True,
            collate_fn=self.collate_fn
        )

        self.logger.write(
            "Number of training samples: {}".format(self.train_dataset.__len__())
        )
        

    def build_model(self):
        self.logger.write("Building model")
        model_name = self.config.model_config.model_name

        if model_name == "muscall":
            self.model = MusCALL(self.config.model_config)
        else:
            raise ValueError("{} model is not supported.".format(model_name))

        self.print_parameters()

        self.model.to(self.device)

    def build_optimizer(self):
        self.logger.write("Building optimizer")
        optimizer_config = self.config.training.optimizer
        self.optimizer = getattr(optim, optimizer_config.name, None)(
            self.model.parameters(), **optimizer_config.args
        )

        num_train_optimization_steps = (
            int(self.train_loader.dataset.__len__() / self.batch_size)
            * self.config.training.epochs
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_train_optimization_steps * 0.1, eta_min=1e-6,
        )

    def get_sentence_similarities(self, data_loader, data_idx):
        raw_captions = [
            data_loader.dataset.get_raw_caption(idx.item()) for idx in data_idx
        ]
        sentence_embeddings = self.sbert_model.encode(
            raw_captions,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return sentence_embeddings @ sentence_embeddings.t()

    def train(self):
        best_val_loss = 10000

        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write(
                "Resumed training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            
            self.load_ckp(self.logger.checkpoint_path)
            self.logger.write(
                f"Resumed training from epoch {self.start_epoch} with best val loss {best_val_loss}"
            )
        else:
            self.logger.write(
                "Started training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.start_epoch = 0

        for epoch in range(tqdm(self.start_epoch, self.config.training.epochs)):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(self.train_loader, is_training=True)
            mlflow.log_metric("avg_train_loss", train_loss, step=epoch + 1)

            val_loss = self.train_epoch_val(self.val_loader)
            mlflow.log_metric("avg_val_loss", val_loss, step=epoch + 1)

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            # save checkpoint in appropriate path (new or best)
            self.logger.save_checkpoint(state=checkpoint, is_best=is_best)

    def load_ckp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"]

    def train_epoch(self, data_loader, is_training=False):
        running_loss = 0.0
        n_batches = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        for i, batch in enumerate(tqdm(data_loader, desc="Processing batches")):                     
            original_mel_spectograms = batch["input_audio"]
            text_input_ids = batch["text_input_ids"]
            text_attention_mask = batch["text_attention_mask"]
            data_idx = batch["idx"]
            original_audio = batch["original_audio"]

            if self.config.model_config.loss == "weighted_clip":
                sentence_sim = self.get_sentence_similarities(data_loader, data_idx)
            else:
                sentence_sim = None

            # Cast operations to mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.training.amp):
                loss = self.model(
                    original_mel_spectograms,
                    text_input_ids,
                    original_audio=original_audio,
                    sentence_sim=sentence_sim,
                    text_mask=text_attention_mask
                )

            if is_training:
                if self.config.training.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # clamp temperature scaling if over log(100)
                if self.model.logit_scale.item() > np.log(100):
                    self.model.logit_scale.data = torch.clamp(
                        self.model.logit_scale.data, max=np.log(100)
                    )

                self.scheduler.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()
            n_batches += 1

            # Log metrics to MLflow 
            if is_training:
                self.global_step += 1
                avg_loss = loss.item() / self.batch_size
                learning_rate = self.scheduler.get_last_lr()[0]
                mlflow.log_metric("training_loss", avg_loss, step=self.global_step)
                mlflow.log_metric("learning_rate", learning_rate, step=self.global_step)

        return running_loss / n_batches

    def train_epoch_val(self, data_loader):
        with torch.no_grad():
            loss = self.train_epoch(data_loader, is_training=False)
        return loss
    
    
"""
Bridge the gap between modalities and learn joint representations.

Design choice: 
* Kan det bli problem med att sentence bert mappar till 768 embedding size och whisper till 1024?
* What type of tokenization? (Character level, Word level, Subword level (byte pair encoding))
* Kan det bli problem med kb swedish bert cased - kompensera för det i datasettet
* För syftet entity recogntion - Vilken text encoder bör man använda? Vilket dataset bör man använda? Sträcker sig data representativ för fallet hela vägen hit? 
* Kan man använda KTH GPU och träna filtering framework modellen redan nu
* Kan det ta bort syftet med detta att KBs whisper redan är bra 
* Hur länge ska man träna? 
* Spotify podcast dataset

# Om träning på KTH
- Får ett konto 
- Då kan man logga in remote och köra på deras GPU
- Vanliga UBUNTU maskiner 
- Temux bra kommando att komma ihåg, blir en virtuell terminal som man kan logga ut ifrån


Authors curated three datasets: MLS, Common voice and PSFB  
Initial fine-tuning of a Whisper Small model on the combined real and synthetic data yielded unsatisfactory results.
To improve they developed a novel filtering methodology inspired by MusCALL
They used Common Voice portuguese
Each transcription is embedded by a DeBERTa based model pretrained on portuguese text
Each audio is embedded by a pretrained Whisper-medium model

The pre-trained encoders are frozen during training,
The focus is on optimizing the parameters of the projection layers

Traditional approaches assume that all items inside of a dataset are perfectly aligned, with each audio-text pair representing the most semantically relevant match, 
and that all non-aligned pairs are equally dissimilar. However, in real-world datasets within a randomly sampled mini-batch, some samples will share similarities with other tracks

This filtering process generated three distinct synthetic datasets of different sizes, each reflecting a different similarity threshold. 
These filtered datasets were then combined with three real speech datasets (MLS, CV, and PSFB) to create three augmented training sets.

In this research, they generated 48972 utterances of synthetic data, approximately 140 hours
The text data for TTS was sourced from the CAPES dataset.
 
Following the filtering of synthetic data and its integration with the real speech datasets, 
a thorough examination of the resulting datasets was conducted.

"""