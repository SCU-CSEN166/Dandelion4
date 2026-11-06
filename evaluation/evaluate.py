import json

# Read expected labels
with open("examples/expected_outputs.json") as f:
    expected = json.load(f)

# Read baseline predictions
with open("examples/baseline_output.json") as f:
    predicted = json.load(f)

# Match results by message ID
predictions = {
    record["id"]: record
    for record in predicted
}

correct = 0
errors = []

for record in expected:
    actual = predictions.get(record["id"])

    if actual is None:
        errors.append(record["id"])
        continue

    # Both relevance and activity type must match
    if (
        record["relevant"] == actual["relevant"]
        and record["activity_type"] == actual["activity_type"]
    ):
        correct += 1
    else:
        errors.append(record["id"])

accuracy = correct / len(expected) if expected else 0

print("Evaluation complete!")
print(f"Correct classifications: {correct}/{len(expected)}")
print(f"Accuracy: {accuracy:.2%}")
print("Incorrect message IDs:", errors)