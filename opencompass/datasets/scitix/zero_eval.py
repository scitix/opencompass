import json
import re

# adapted from https://github.com/WildEval/ZeroEval/blob/8c1485edf12c6efb5f69135a562927c5ad484059/src/templates/MCQA.py
# Multiple Choice Question Answer Template
MCQA = """
## Question: 

{question}

## Choices:

{choices}

## Instruction 

Please answer this question by first reasoning and then selecting the correct choice. 
Present your reasoning and solution in the following json format. 
Please show your choice in the `answer` field with only the choice letter, e.g.,`"answer": "C"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```
"""


# adapted from https://github.com/WildEval/ZeroEval/blob/8c1485edf12c6efb5f69135a562927c5ad484059/src/evaluation/eval_utils.py
def generate_choice_string(choices):
    choice_string = ""
    for i, choice in enumerate(choices):
        choice_string += f"- ({chr(65 + i)}) {choice}\n"
    return choice_string


def extract_values_from_json(
    json_string, keys=["reasoning", "answer"], allow_no_quotes=False
):
    extracted_values = {}
    for key in keys:
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f"{key}\\s*:\\s*([^,\\s]*)"
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def extract_first_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                _ = stack.pop()
                if not stack:
                    # Complete JSON object found
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError:
                        return None
                    finally:
                        first_json_start = None
    return None
