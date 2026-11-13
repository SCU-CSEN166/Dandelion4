from transformers import pipeline
from pathlib import Path
import json

# Load a small pretrained Hugging Face model
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    device="cpu"
)

# Generate fictional WhatsApp field activity messages
prompt = """
Generate 10 short fictional WhatsApp messages from
community field workers in Kenya.

Include reports about:
- Household visits
- Health screenings
- Clinic referrals
- Women's community meetings
- Unrelated everyday conversations

Use different writing styles and natural English.
Do not include real personal information.

Write exactly one message per line.
Do not include numbering or explanations.
"""

messages = [
    {"role": "user", "content": prompt}
]

result = generator(
    messages,
    max_new_tokens=700,
    do_sample=True,
    temperature=0.7
)

# Extract generated text
output = result[0]["generated_text"][-1]["content"]

# Convert generated lines into JSON records
lines = [
    line.strip()
    for line in output.splitlines()
    if line.strip()
]

dataset = []

for i, line in enumerate(lines):
    dataset.append({
        "id": f"msg_{i+1:03d}",
        "sender_id": f"worker_{(i % 5)+1:02d}",
        "message": line
    })

# Save dataset
Path("examples").mkdir(exist_ok=True)

with open("examples/huggingface_messages.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} messages!")
print("Saved to examples/huggingface_messages.json")