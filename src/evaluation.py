from src.generate_responses import format_question, contains_answer
from src.config import EVALUATION_TEMPERATURE, EVALUATION_TOP_P

def chat(model, tokenizer, user_query, system_message:str="", t:float=EVALUATION_TEMPERATURE, tp:float=EVALUATION_TOP_P) -> str:
    """
    Interface for LLM chat. Returns the LLM's response as a string.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=t,
        top_p=tp
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def generate_user_message(list1:list, list2:list) -> str:
    """
    Given two lists, returns a formatted user message for the steganalyst.
    """
    return f"First set of responses:\n```\n{list1}\n```\n\nSecond set of responses:\n```\n{list2}\n```"