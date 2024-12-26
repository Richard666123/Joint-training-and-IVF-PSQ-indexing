import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import csv

file_path = 'DATA-19.xlsx'
df = pd.read_excel(file_path, header=None)
questions = df[0].tolist()
model_path = './COMPNet'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
    attention_mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(token_embeddings * attention_mask, 1)
    count = torch.sum(attention_mask, 1)
    return summed / count

embeddings = []
for question in questions:
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    pooled_output = mean_pooling(outputs, inputs['attention_mask'])

    pooled_output = pooled_output.squeeze(0)

    max_val = pooled_output.max().item()
    min_val = pooled_output.min().item()
    normalized_output = (pooled_output - min_val) / (max_val - min_val)

    embeddings.append(normalized_output.numpy())

output_file = 'embedding.csv'
header = [f"dim_{i}" for i in range(768)]
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for embedding in embeddings:
        writer.writerow(embedding)
