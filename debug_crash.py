from openai import OpenAI

client = OpenAI()

file_id = "file-9WPZweX4v1pSyyS63ZmpDE"

matches = [batch for batch in client.batches.list(limit=100) if batch.input_file_id == file_id]

for batch in matches:
    print(batch.id, batch.status, batch.created_at)
