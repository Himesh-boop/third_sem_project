import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Freud/freud_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto"
)

model.eval()

def test_freud(user_input):
    prompt = (
        "SYSTEM: You are Freud, an empathetic mental health companion.\n"
        f"User: {user_input}\n"
        "Assistant:\n<ANSWER>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

# Tests
tests = [
    "hi",
    "I can't sleep",
    "I feel alone"
]

for msg in tests:
    print(f"User: {msg}")
    print(f"Freud: {test_freud(msg)}\n")