import pandas as pd
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_not_exception_type
from typing import List, Dict
from pydantic import BaseModel
from time import time
from openai.types import CompletionUsage
import os 
import re
import json
import asyncio 
from dotenv import load_dotenv
from jiwer import cer as jiwer_cer

file = "missed_entities_not_norm_entities_benchmark_cv_small.csv"
filepath = f"../results/{file}"

df = pd.read_csv(filepath)
print(df.head())

NUM_ROWS = 40
NUM_MAX_RETRY_ATTEMPTS = 2
MAX_COMPLETION_TOKENS = 3_000
PROMPT_TEMPLATE = (
    "You are given a ground truth sentence (Ground truth), a candidate sentence (candidate) outputted from whisper, and an extracted entity (entity) from the ground truth (Ground truth).\n"
    "Your task is to extract the attempt at transcribing the entity from the candidate sentence \n"
    "Do NOT correct the entity, just extract it.\n"
    "Example: Candidate: Ni heter Lebauski och jag heter Lebauski, Ground truth: Så ni heter Lebowski, och jag heter Lebowski., Entity: Lebowski. In this case you return Lebauski, as this is the candidates attempt at the entity.\n"
    "ONLY return the extracted entity in plain JSON:\n"
    "{{\"extracted_entity\": \"entity1\"}}\n\n"
    "Ground truth: {ground_truth}\n"
    "Candidate: {candidate}\n"
    "Entity: {entity}\n\n"
    "Extracted JSON:"
)

class GPTResponse(BaseModel):
    extracted_entity: str
    
class InitialFilterResponseFormat(BaseModel):
    plan: str  # Brief summary of the classification approach
    summary: str  # Short summary of the text
    reasoning: str  # Explanation with evidence (quotes, phrases) supporting the decision
    is_relevant: bool  # Whether the content is relevant or not


load_dotenv()
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
        pass

    @retry(
        stop=stop_after_attempt(num_max_retry_attempts),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type(CompletionUsage),
        before=_before_retry,
        after=_after_retry
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


messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(
    ground_truth=row['ground_truth'],
    candidate=row['candidate'],
    entity=row['entity']
)} for _, row in df.iloc[:NUM_ROWS].iterrows()]

extracted_entities =[]

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

df_corrected = df.iloc[:NUM_ROWS].copy()

batch_size = 500
for start_idx in range(0, NUM_ROWS, batch_size):
    print(f"Processing batch {start_idx} to {min(start_idx + batch_size, NUM_ROWS)}")
    end_idx = min(start_idx + batch_size, NUM_ROWS)
    batch_messages = messages[start_idx:end_idx]
    responses = asyncio.run(gather_responses(batch_messages, GPTResponse, 0.0, "gpt-4o-mini"))
    
    for idx, corrected in enumerate(responses):
        df_corrected.at[start_idx + idx, 'cand_entity'] = corrected.extracted_entity
        
    print(f"Completed batch {start_idx} to {end_idx}")
    

# Calculate the CER for each row
df_corrected['cer'] = df_corrected.apply(
    lambda row: jiwer_cer(row['entity'], row['cand_entity']), axis=1
)

# Print the CER for each row
    
total_cer = df_corrected['cer'].sum() / len(df_corrected)
print(f"Total average CER: {total_cer}")
    
df_corrected.to_csv(f"../cer_results/cer_{file}", index=False)

