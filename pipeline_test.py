query = "punishment of murder"

# Step 1: Classify Act
act, conf = classify_act(query)
print(f"Act: {act}, Confidence: {conf:.2f}")

# Step 2: Retrieve Top-k Docs
docs = retrieve_context(query, act=act if conf>0.7 else None)

# Step 3: Generate Answer
answer = generate_answer(query, docs)
print("\nFinal Answer:\n", answer)
