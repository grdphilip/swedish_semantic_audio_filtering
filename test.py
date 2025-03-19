from datasets import load_dataset
from datasets import concatenate_datasets

# Load the Fleurs dataset and filter for Swedish (sv_se)
fleurs_df = load_dataset("google/fleurs", "sv_se")
common_voice_df = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE")
dataset = load_dataset("KTH/nst", "speech")


fleurs_train = fleurs_df['train']
fleurs_val = fleurs_df['validation']
fleurs_test = fleurs_df['test']
# Print available splits

common_voice_train = common_voice_df['train']
common_voice_val = common_voice_df['validation']
common_voice_test = common_voice_df['test']


cv_concat = concatenate_datasets([common_voice_train, common_voice_val, common_voice_test])
fleurs_concat = concatenate_datasets([fleurs_train, fleurs_val, fleurs_test])

print(len(cv_concat))
print(len(fleurs_concat))