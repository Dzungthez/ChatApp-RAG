import re

pattern = r"Answer:\s*(.*)"  # regex pattern to extract the answer from the model response


def extract_answer_from_model(model_response: str):
    """
    Extracts the answer from the model response
    """
    match = re.search(pattern, model_response)
    if match:
        answer = match.group(1).strip()
    else:
        answer = "No answer found in model response."
    return answer
