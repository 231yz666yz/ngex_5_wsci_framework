from pathlib import Path
from ollama import chat
import json



question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""


## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the kyeword "print" or "printer", then the function should return the file "knowledge/printer_setup.txt" in a list.
def select_context(question):
    keywords = {
        "wi-fi": "wifi_setup",
        "wifi": "wifi_setup",
        "password": "password_changes",
        "email": "email_setup",
        "print": "printing",
        "printer": "printing",
        "status": "service_status",
        "vpn": "vpn",
        "projector": "classroom_projectors",
        "classroom": "classroom_projectors"
    }

    q = question.lower()

    names = []
    for keyword, name in keywords.items():
        if keyword in q:
            if name not in names:
                names.append(name)

    names.sort()

    files = []
    for n in names:
        files.append(f"knowledge/{n}.txt")

    return files

selected_files = select_context(question)

## READ SELECTED FILES and add their contents to the context variable.
context = ""

for file in selected_files:
    context += Path(file).read_text()
    context += "\n\n"

print("Context characters (selected):", len(context))


## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context"

def compress_context(context, question):
    response = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only the information relevant to the question "
                    "from the context supplied by the user. Keep it short."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    return response.message.content


## Print the length of the compressed context
compressed_context = compress_context(context, question)
print(len(compressed_context))

## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output

## WRITE ##
response = chat(
    model="qwen3:8b",
    format="json",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer student IT-support questions using only the context "
                "supplied by the user. Never invent details. "
                "Reply as JSON with exactly these keys: "
                "device, cause, action, wifi_status."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{compressed_context}\n\nQuestion:\n{question}"
        }
    ]
)

print(response.message.content)

## WRITE the above output in an artifact called "state"
state = json.loads(response.message.content)

with open("state.json", "w") as file:
    json.dump(state, file, indent=2)

## Update the rest of the code so that it uses the "state" artifact as part of the context.
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.

## ISOLATE ##
diagnostic_context = {
    "problem": question,
    "device": state["device"],
    "wifi_status": state["wifi_status"]
}

report_context = {
    "total_wifi_cases": 37,
    "resolved_cases": 29,
    "unresolved_cases": 8
}

classification = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "Classify the problem supplied by the user as either "
                "'diagnostic' or 'report'. Reply with one word only."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

if "report" in classification.message.content.lower():
    selected_context = report_context
else:
    selected_context = diagnostic_context

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": f"Context:\n{json.dumps(selected_context, indent=2)}\n\nQuestion:\n{question}"
        }
    ]
)

print(response.message.content)
