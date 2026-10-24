LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def format_demonstration(text, label):
    label_name = LABEL_NAMES[int(label)]

    return (
        f"Article:\n{text}\n"
        f"Category: {label_name}\n"
    )


def build_icl_prompt(demonstrations, query_text):
    prompt = """You are an AG-News text classification system.

Classify the query article into exactly one of these categories:

World
Sports
Business
Sci/Tech

Here are some examples:

"""

    for text, label in demonstrations:
        prompt += format_demonstration(text, label)
        prompt += "\n"

    prompt += f"""Now classify this article:

Article:
{query_text}

Answer with only one category:
World
Sports
Business
Sci/Tech
"""

    return prompt