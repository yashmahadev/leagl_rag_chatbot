from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Llama 3 or Mistral model (CPU or GPU)
model_name = "TheBloke/Llama-3-7B-Instruct-GPTQ"  # example
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Generate answer
def generate_answer(query, context_docs):
    context_text = "\n".join(context_docs)
    prompt = f"""
You are an Indian legal assistant.
Answer the question using ONLY the context below.
Cite Act + Section.
If not found, reply: 'No relevant section found.'

Context:
{context_text}

Question: {query}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Test
query = "punishment of murder"
docs = retrieve_context(query, act="IPC")
answer = generate_answer(query, docs)
print(answer)
