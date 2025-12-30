import numpy as np
import pandas as pd
import json
import random

with open("Dataset.json", "r") as f:
    data = json.load(f)
    sample_data = []
    skipped_intents = []
    
    for intent in data["intents"]:
        tag = intent.get("tag", "unknown")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses")

        if not responses:
            skipped_intents.append(tag)
            continue
        
        for pattern in patterns:
            response = random.choice(responses)
            sample = f'''"text": """User: {pattern}\\nAssistant: {response}""", '''
            sample_data.append(sample)
            
with open("PreProcessed_data.txt", "w") as f:
    for s in sample_data:
        f.write(s+"\n\n")

print(f"Generated: {len(sample_data)} samples")
print(f"Skipped: {len(skipped_intents)} samples")