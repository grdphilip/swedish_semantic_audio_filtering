from datasets import Dataset, load_dataset
from huggingface_hub import list_datasets, login
from dotenv import load_dotenv
import logging
import os
import json
import openai
from openai import OpenAI, AsyncOpenAI
import re
from datasets import Audio, Features, Value, Sequence
import asyncio 
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_not_exception_type
from typing import List, Dict
from pydantic import BaseModel
from time import time
from openai.types import CompletionUsage



NUM_MAX_RETRY_ATTEMPTS = 2
MAX_COMPLETION_TOKENS = 3_000
NUM_ROWS = None

load_dotenv()
token = os.getenv('HF_WRITE')
os.environ['HF_TOKEN'] = token
login(token=token)

# datasets = list_datasets()
# print([d.id for d in datasets if "grdphilip" in d.id])

dataset = "cv"
update_path = f"{dataset}_corrected"



dataset_map: dict = {
    "cv": "grdphilip/cv_swedish_with_entities",
    "fleurs": "grdphilip/fleurs_swedish_with_entities",
    "cv_corrected": "grdphilip/cv_swedish_with_entities_corrected",
    "fleurs_corrected": "grdphilip/fleurs_swedish_with_entities_corrected",
}

df = load_dataset(dataset_map[dataset])['train']
api_key = os.getenv("TOMAS_OPENAI_KEY")
client = AsyncOpenAI(api_key=api_key)

async def get_responses(response_format: BaseModel,
    messages: List[Dict[str, str]],
    temperature: float,
    model: str = "gpt-4o-mini",
    num_max_retry_attempts: int = NUM_MAX_RETRY_ATTEMPTS,
) -> List[BaseModel]:
    
    """
        Fetches responses from the OpenAI API with retry logic.

        Parameters:
            response_format (GPTResponse): The format to parse the GPT response into.
            prompt (str): The prompt to send to the OpenAI API.
            temperature (float): The temperature setting for the API.
            num_completions (int): The number of completions to request.
            model (str): The model to use for the API request.
            fn_retry_if_not_valid (Optional[Callable[[List[GPTResponse]], bool]]): A function that validates the responses and triggers a retry if needed.
            num_max_retry_attempts (int): Maximum number of retry attempts.
    """
    def _before_retry(state):
        pass

    def _after_retry(state):
        print(f"Retry info after {state.attempt_number} attempts")
        
    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(num_max_retry_attempts),
        before=_before_retry,
        after=_after_retry,
        retry_error_callback=lambda retry_state: retry_state.outcome.result() if retry_state.outcome.failed else retry_state.outcome.result()
    )
    
    async def _make_request():
        start_time = time()
        responses = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_COMPLETION_TOKENS,
            n=1
        )
        
        try:
            completion_usage = CompletionUsage.model_validate(responses.usage)
            choices = responses.choices
            finish_reasons = [choice.finish_reason for choice in choices]
            content = [choice.message.content for choice in choices]
        except Exception as e:
            print("Error parsing response:", e)
            raise e
        
        try:
            json_content = []
            for c in content:
                cleaned_content = re.sub(r'^```json|```$', '', c.strip(), flags=re.MULTILINE).strip()
                json_content.append(json.loads(cleaned_content))

        except json.JSONDecodeError as e:
            print("Error repairing JSON:", e, "Content was:", content)
            raise e


        try:
            parsed_responses = [response_format.model_validate(c) for c in json_content]
        except Exception as e:
            print("Error validating response:", e)
            raise e

        return parsed_responses
    
    return await _make_request()
    
    
async def indexed_get_response(i, messages, response_format, temperature, model):
    result = await get_responses(
        response_format=response_format,
        messages=messages,
        temperature=temperature,
        model=model,
    )
    return i, result[0]
    
async def get_initial_filter_responses(initial_docs, search_profile_id, temperature, gpt_model):
    with open(f"search_profiles/{search_profile_id}/prompts/initial_filter_inclusive.md", "r") as file:
        system_prompt = file.read()

    tasks = []
    for i, doc in enumerate(initial_docs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": doc.text},
        ]
        tasks.append(asyncio.create_task(indexed_get_response(i, messages, InitialFilterResponseFormat, temperature, gpt_model)))

    responses = [None] * len(tasks)
    completed_count = 0
    ten_percent = max(int(0.1*len(initial_docs)), 1)
    start_time = time()
    for future in asyncio.as_completed(tasks):
        index, result = await future
        responses[index] = result  # preserve order
        completed_count += 1
        if completed_count % ten_percent == 0:
            print(f"Completed {completed_count}/{len(initial_docs)} requests in {time() - start_time:.2f} seconds")

    print(f"Completed All requests in {time() - start_time:.2f} seconds")
    return responses
    
prompt_template = (
    "You are given a sentence (text), extracted entities, and their metadata. "
    "Your task is ONLY to fix obvious formatting errors in entities and metadata:\n"
    "- Merge entities mistakenly split.\n"
    "- Separate entities mistakenly joined.\n"
    "- Ensure entities EXACTLY match substrings from the original text.\n"
    "- DO NOT add additional information from the text about an entity that is not already present in the entity.\n"
    "- ALL text in the reformatted entity MUST be present in the original entity.\n"
    "- ALL entities must EXACTLY match the substrings as they appear in the original text, even if grammatically incorrect.\n"
    "\n"
    "Return ONLY corrected entities and metadata in plain JSON:\n"
    "{{\"entities\": [\"entity1\", \"entity2\"], \"metadata\": {{\"entity\": [\"entity1\", \"entity2\"], \"entity_type\": [\"type1\", \"type2\"]}}}}\n\n"
    "Text: {text}\n"
    "Entities: {entities}\n"
    "Metadata: {metadata}\n\n"
    "Corrected JSON:"
)

if NUM_ROWS == None: NUM_ROWS = len(df)
print(f"Processing {NUM_ROWS} rows")

messages = [{"role": "user", "content": prompt_template.format(text=row['text'], entities=row['entities'], metadata=row['metadata'])} for row in df.select(range(NUM_ROWS))]

class InitialFilterResponseFormat(BaseModel):
    plan: str  # Brief summary of the classification approach
    summary: str  # Short summary of the text
    reasoning: str  # Explanation with evidence (quotes, phrases) supporting the decision
    is_relevant: bool  # Whether the content is relevant or not
    
class CorrectedEntities(BaseModel):
    entities: List[str]
    metadata: Dict[str, List[str]]
    
corrected_rows = []
async def gather_responses(messages, response_format, temperature, model):
    tasks = [
        indexed_get_response(i, [message], response_format, temperature, model) 
        for i, message in enumerate(messages)
    ]
    responses = [None] * len(tasks)
    completed_count = 0
    ten_percent = max(int(0.1 * len(messages)), 1)
    start_time = time()

    for future in asyncio.as_completed(tasks):
        index, result = await future
        responses[index] = result  # preserve order
        completed_count += 1
        if completed_count % ten_percent == 0:
            print(f"Completed {completed_count}/{len(messages)} requests in {time() - start_time:.2f} seconds")

    print(f"Completed All requests in {time() - start_time:.2f} seconds")
    return responses

df_corrected = df.select(range(NUM_ROWS)).to_pandas()

batch_size = 500
for start_idx in range(0, NUM_ROWS, batch_size):
    print(f"Processing batch {start_idx} to {min(start_idx + batch_size, NUM_ROWS)}")
    end_idx = min(start_idx + batch_size, NUM_ROWS)
    batch_messages = messages[start_idx:end_idx]
    responses = asyncio.run(gather_responses(batch_messages, CorrectedEntities, 0.0, "gpt-4o-mini"))

    for idx, corrected in enumerate(responses):
        df_corrected.at[start_idx + idx, 'entities'] = corrected.entities
        df_corrected.at[start_idx + idx, 'metadata'] = corrected.metadata
        
    print(f"Completed batch {start_idx} to {end_idx}")

for i, row in df_corrected.iterrows():
    print(f"Row {i}: Text: {row['text']}")
    print(f"Row {i}: Original Entities: {df[i]['entities']}")
    print(f"Row {i}: Corrected Entities: {row['entities']}")



features = Features({
    'audio': Audio(sampling_rate=16000),
    'path': Value('string'),
    'text': Value('string'),
    'entities': Sequence(Value('string')),
    'metadata': Sequence({
        'entity': Value('string'),
        'entity_type': Value('string')
    })
})


manifest_dict = {
    'audio': df_corrected['audio'].tolist(),
    'path': df_corrected['path'].tolist(),
    'text': df_corrected['text'].tolist(),
    'entities': df_corrected['entities'].tolist(),
    'metadata': df_corrected['metadata'].tolist()
}

corrected_dataset = Dataset.from_dict(manifest_dict, features=features)
corrected_dataset.push_to_hub(dataset_map[update_path])
print(corrected_dataset)
print(f"Pushed to {dataset_map[update_path]}")





