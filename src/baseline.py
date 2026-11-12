import json
import re
import sys
from pathlib import Path

# Keywords for different Dandelion Africa activities
KEYWORDS = {
    "household_outreach": [
        "visited", "households", "home visits",
        "reached families"
    ],
    "health_screening": [
        "screened", "screening", "health camp"
    ],
    "community_meeting": [
        "held a meeting", "community meeting",
        "workshop", "organized a meeting"
    ],
    "health_referral": [
        "referred", "referrals", "referral"
    ]
}

LOCATIONS = ["Nakuru", "Baringo", "Narok", "Kajiado"]

# Remove phone-number patterns before processing.
# This is only a basic anonymization method.
def anonymize(text):
    return re.sub(
        r"(?<!\w)(?:\+?254|0)[\s-]?[17]"
        r"(?:[\s-]?\d){8}(?!\w)",
        "[PHONE]",
        text
    )


# Find the activity described in a message
def classify_message(text):
    text = text.lower()

    for activity, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return activity

    return None


# Find a location mentioned in the message
def extract_location(text):
    for location in LOCATIONS:
        if re.search(
            r"\b" + re.escape(location) + r"\b",
            text,
            re.IGNORECASE
        ):
            return location

    return None


# Find a number related to people or households
def extract_count(text):
    match = re.search(
        r"\b(\d+)\s+"
        r"(households?|families|women|people|participants)"
        r"\b",
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


# Process one WhatsApp message
def process_message(message):
    cleaned = anonymize(message["message"])
    activity = classify_message(cleaned)

    return {
        "id": message["id"],
        "relevant": activity is not None,
        "activity_type": activity,
        "location": extract_location(cleaned),
        "reported_count": extract_count(cleaned),
        "verification_status": (
            "pending" if activity else "not_applicable"
        )
    }


# Choose input file from command line
input_file = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "examples/synthetic_messages.json"
)

# Load the synthetic WhatsApp messages
with open(input_file, "r", encoding="utf-8") as f:
    messages = json.load(f)

# Process every message
results = []

for message in messages:
    record = process_message(message)
    results.append(record)

# Save structured JSON output
Path("examples").mkdir(exist_ok=True)

output_file = "examples/baseline_output.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Show results in terminal
print("Baseline completed!")
print(f"Processed {len(results)} messages.")

relevant_count = sum(
    1 for record in results if record["relevant"]
)

print(f"Relevant messages: {relevant_count}")
print(f"Results saved to {output_file}")

# Display first two outputs
print("\nExample results:")
print(json.dumps(results[:2], indent=2))