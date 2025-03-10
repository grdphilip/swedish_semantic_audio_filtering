#!/bin/bash


python process_hf_dataset.py grdphilip/elevenlabs_syndata default train customized syndata_11labs_train_manifest.json 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


python process_data_for_filtering.py syndata_11labs_train_manifest.json ./

python apply_filering.py data/syndata_11labs_train_manifest.json