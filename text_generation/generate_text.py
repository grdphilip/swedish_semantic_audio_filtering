from openai import OpenAI
import pandas as pd
#from transformers import pipeline
from dotenv import load_dotenv
import os
#from transformers import AutoModel,AutoTokenizer

# Load API key from .env file
def get_credentials():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key   

# num completions 

api_key = get_credentials()
client = OpenAI(api_key=api_key)
# tok = AutoTokenizer.from_pretrained('KBLab/bert-base-swedish-cased')
# model = AutoModel.from_pretrained('KBLab/bert-base-swedish-cased')

# Generate multiple Swedish sentences
def clean_sentence(sentence):
    # Remove quotation marks and similar characters
    cleaned = sentence.replace('"', '').replace("'", "")
    return cleaned.strip()

def generate_swedish_sentences(num_samples):
    sentences = []
    for _ in range(num_samples):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Only return the Swedish sentence. Nothing more, nothing less."},
                {
                    "role": "user",
                    "content": "Generera en svensk mening innehållandes en eller flera entiteter (Person, Organization, Plats etc). Språket bör vara spontan-tal"
                }
            ]
        )
        # Extract the message part of the completion
        raw_sentence = completion.choices[0].message.content
        cleaned_sentence = clean_sentence(raw_sentence)  # Clean the sentence to remove unwanted characters
        print(cleaned_sentence)  # Optional: print the cleaned sentence
        sentences.append(cleaned_sentence)
        
    return sentences

num_sentences = 100
swedish_sentences = generate_swedish_sentences(num_sentences)

# Function to extract entities using a Swedish NER model
# def extract_entities(texts):
    
#     entities = []
#     for text in texts:
#         print(f"Extracting entities from: {text}")
#         ner_results = nlp(text)
#         extracted = [{ent['entity_group']: ent['word']} for ent in ner_results]
#         entities.append(extracted)
#     return entities

# # Generate synthetic data
# entities_extracted = extract_entities(swedish_sentences)

# Save to DataFrame and then to CSV
df = pd.DataFrame({
    'sentence': swedish_sentences,
})
df.to_csv('./dataset/syndata.csv', index=False)

print("Data generation complete. The syndata.csv file has been created with Swedish sentences and their entities.")
