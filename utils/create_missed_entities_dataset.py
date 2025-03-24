import pandas as pd
import random
import ast 
import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict
import asyncio  
from time import time


# =================== PROCESSING MISSED ENTITIES  ===================

missed_entities_cv = pd.read_csv('../results/missed_entities_not_norm_entities_benchmark_cv_small.csv')
missed_entities_fleurs = pd.read_csv('../results/missed_entities_not_norm_entities_benchmark_fleurs_small.csv')
sentence_pool = pd.read_csv('../results/sentence_pool.csv')

total_missed_entities = pd.concat([missed_entities_cv, missed_entities_fleurs])
total_missed_entities = total_missed_entities.drop_duplicates(subset='entity')
total_missed_entities = total_missed_entities[total_missed_entities['type'].isin(['PER', 'ORG', 'LOC'])]

print(f"Processing {len(total_missed_entities)} missed entities")

# =================== HELP METHODS ===================

def parse_metadata(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)  # Converts the string representation into a Python object
        except:
            return []
    return x

def sample_sentences_by_entity_type(buckets, entity_type, n=10):
    samples_container = []
    sampled = random.sample(buckets[entity_type], n)
    sampled_indices = [row['index'] for row in sampled]  # keep track of original indices
    
    samples_container = [row['sentence'] for row in sampled]
    return samples_container, sampled_indices


# =================== PROCESSING SENTENCE POOL  ===================

sentence_pool.reset_index(inplace=True)
sentence_pool['metadata'] = sentence_pool['metadata'].apply(parse_metadata)
print(type(sentence_pool.iloc[0]['metadata'])) 

sentence_pool['entity_type'] = sentence_pool['metadata'].apply(
    lambda items: items[0].get('entity_type') 
                  if isinstance(items, list) and items and isinstance(items[0], dict) 
                  else None
)

print(sentence_pool[['metadata', 'entity_type', 'sentence']].head())


buckets = {
    'PER': sentence_pool[sentence_pool['entity_type'] == 'PER'].to_dict('records'),
    'ORG': sentence_pool[sentence_pool['entity_type'] == 'ORG'].to_dict('records'),
    'LOC': sentence_pool[sentence_pool['entity_type'] == 'LOC'].to_dict('records'),

}

print("Total entities in each bucket:") 
for key, val in buckets.items():
    print(key, len(val))
    
    
load_dotenv()
api_key = os.getenv("TOMAS_OPENAI_KEY")
client = AsyncOpenAI(api_key=api_key)


PROMPT_TEMPLATE = (
        "You are given candidate sentences, each containing an entity of type '{entity_type}'.\n"
        "Your task is to select the sentence that best fits incorporating the missed entity '{missed_entity}'.\n"
        "Then, modify that sentence by replacing its original entity with '{missed_entity}', ensuring the sentence remains grammatically correct.\n"
        "\n"
        "ONLY RETURN the modified sentence.\n"
        "\n"
        "Candidate sentences:\n"
        "{candidate_sentences}\n"
    )

async def generate_corrected_sentence(
    client: AsyncOpenAI,
    candidate_sentences: List[str],
    entity_type: str,
    missed_entity: str,
    prompt_template: str,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini"
):
    prompt = prompt_template.format(
        entity_type=entity_type,
        missed_entity=missed_entity,
        candidate_sentences="\n".join(f"{i}. {s}" for i, s in enumerate(candidate_sentences, 1))
    )
    # You need an async OpenAI call here, e.g. client.chat.completions.acreate
    # if your OpenAI wrapper supports it. If not, you'd do an async request manually.
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    corrected_sentence = response.choices[0].message.content.strip()
    return {
        "sentence": corrected_sentence,
        "entity": missed_entity,
        "entity_type": entity_type
    }
    
async def gather_responses(
client: AsyncOpenAI,
tasks_data: List[Dict],
prompt_template: str,
concurrency: int = 500
):

    sem = asyncio.Semaphore(concurrency)

    async def run_task(i, data):
        async with sem:
            result = await generate_corrected_sentence(
                client=client,
                candidate_sentences=data["candidate_sentences"],
                entity_type=data["entity_type"],
                missed_entity=data["missed_entity"],
                prompt_template=prompt_template
            )
            return i, result

    futures = [run_task(i, data) for i, data in enumerate(tasks_data)]
    results = [None] * len(futures)

    completed = 0
    ten_percent = max(int(0.1 * len(tasks_data)), 1)
    start_time = time()

    for coro in asyncio.as_completed(futures):
        i, result = await coro
        results[i] = result
        completed += 1
        if completed % ten_percent == 0:
            print(f"Completed {completed}/{len(tasks_data)} in {time() - start_time:.2f}s")

    print(f"All {len(tasks_data)} done in {time() - start_time:.2f}s")
    return results

# (C) Example usage
async def process_missed_entities_async(
    client: AsyncOpenAI,
    buckets: Dict[str, List[Dict]],
    total_missed_entities: pd.DataFrame,
    prompt_template: str,
    batch_size: int = 500
):
    # We'll store all new rows here
    new_rows = []

    # Convert DataFrame to list of dicts for iteration
    missed_list = total_missed_entities.to_dict('records')

    # Build "tasks_data" in chunks
    start_idx = 0
    while start_idx < len(missed_list):
        end_idx = min(start_idx + batch_size, len(missed_list))
        batch = missed_list[start_idx:end_idx]
        print(f"Processing batch {start_idx}–{end_idx}")

        tasks_data = []
        for item in batch:
            missed_entity = item['entity']
            entity_type = item['type']

            # Example: For each missed entity, we create 5 tasks
            for _ in range(5):
                candidates = random.sample(buckets[entity_type], 10)
                candidate_sentences = [row["sentence"] for row in candidates]
                
                tasks_data.append({
                    "candidate_sentences": candidate_sentences,
                    "entity_type": entity_type,
                    "missed_entity": missed_entity
                })

        # Gather the async responses for this batch
        results = await gather_responses(
            client=client,
            tasks_data=tasks_data,
            prompt_template=prompt_template,
            concurrency=batch_size  # or any concurrency limit you want
        )

        # Add them to our new_rows list
        for r in results:
            new_rows.append(r)

        start_idx = end_idx
    
    # Convert to DataFrame
    new_df = pd.DataFrame(new_rows)
    return new_df

# (D) Putting it all together
# This is a synchronous "main" function example:
def run_async_correction(client, buckets, total_missed_entities):
    PROMPT_TEMPLATE = (
        "You are given candidate sentences, each containing an entity of type '{entity_type}'.\n"
        "Your task is to select the sentence that best fits incorporating the missed entity '{missed_entity}'.\n"
        "Then, replace the original entity with '{missed_entity}' to maintain grammatical correctness.\n"
        "If replacing the entity results in a contextually incorrect sentence, carefully alter the sentence to ensure contextual correctness, while closely following the original style, structure, and tone.\n"
        "\n"
        "Avoid unrealistic or illogical combinations of organizations, locations, and events (e.g., political parties associating power transfer with unrelated companies, animal welfare incidents attributed to inappropriate organizations like online marketplaces, or historical figures managing farms). Ensure logical coherence, organizational plausibility, geographical accuracy, and factual correctness.\n"
        "\n"
        "NEVER change the missed entity '{missed_entity}' to a different entity.\n"
        "\n"
        "ONLY RETURN the modified, contextually and grammatically correct sentence.\n"
        "\n"
        "Candidate sentences:\n"
        "{candidate_sentences}\n"
    )

    new_dataset = asyncio.run(
        process_missed_entities_async(
            client=client,
            buckets=buckets,
            total_missed_entities=total_missed_entities,
            prompt_template=PROMPT_TEMPLATE,
            batch_size=500
        )
    )

    # Save to CSV
    new_dataset.to_csv("../results/final_sentences.csv", index=False)
    return new_dataset

# Run the async correction

new_dataset = run_async_correction(client, buckets, total_missed_entities)
print(new_dataset.head())
