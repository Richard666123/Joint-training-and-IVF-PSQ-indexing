import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MPNetModel, MPNetTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

data = pd.read_excel('data.xlsx', header=None)

questions = data[0].tolist()
answers = data[1].tolist()

teacher_model_name = 'microsoft/mpnet-base'
teacher_tokenizer = MPNetTokenizer.from_pretrained(teacher_model_name)
teacher_model = MPNetModel.from_pretrained(teacher_model_name)

student_model_name = 'distilbert-base-uncased'
student_tokenizer = DistilBertTokenizer.from_pretrained(student_model_name)
student_model = DistilBertForSequenceClassification.from_pretrained(student_model_name, num_labels=1)

class QA_Dataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = str(self.questions[idx]) if self.questions[idx] is not None else ""
        answer = str(self.answers[idx]) if self.answers[idx] is not None else ""

        encoding = self.tokenizer(question, answer, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        return {key: value.squeeze(0) for key, value in encoding.items()}

def get_cosine_similarity(query, corpus, model, tokenizer):
    query = str(query)
    corpus = [str(ans) for ans in corpus]

    question_encoding = tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    corpus_encoding = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt', max_length=512)

    question_encoding = {key: value.to(model.device) for key, value in question_encoding.items()}
    corpus_encoding = {key: value.to(model.device) for key, value in corpus_encoding.items()}

    question_embeddings = model(**question_encoding)[0][:, 0].detach().cpu().numpy()
    corpus_embeddings = model(**corpus_encoding)[0][:, 0].detach().cpu().numpy()

    question_embeddings = question_embeddings.reshape(1, -1)
    corpus_embeddings = corpus_embeddings.reshape(len(corpus), -1)

    return cosine_similarity(question_embeddings, corpus_embeddings)

def teacher_predict(question, answers, device):
    similarities = get_cosine_similarity(question, answers, teacher_model, teacher_tokenizer)
    max_index = np.argmax(similarities)
    prediction = np.zeros_like(similarities)
    prediction[0, max_index] = 1
    return torch.tensor(prediction, dtype=torch.float32, requires_grad=True).to(device)

def student_predict(question, answers, device):
    similarities = get_cosine_similarity(question, answers, student_model, student_tokenizer)
    max_index = np.argmax(similarities)
    prediction = np.zeros_like(similarities[0])
    prediction[max_index] = 1
    return torch.tensor(prediction, dtype=torch.float32, requires_grad=True).to(device)

def kl_divergence_loss(student_probs, teacher_probs):
    return torch.sum(teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs)))

def train_model(train_loader, teacher_model, student_model, optimizer, device, num_epochs=10):
    student_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100, unit='batch'):
            questions = batch['input_ids'].to(device)
            answers = batch['attention_mask'].to(device)

            teacher_preds = [teacher_predict(q, answers, device) for q in questions]
            student_preds = [student_predict(q, answers, device) for q in questions]

            teacher_probs = torch.stack(teacher_preds).float().to(device)
            student_probs = torch.stack(student_preds).float().to(device)

            soft_loss = kl_divergence_loss(student_probs, teacher_probs)

            hard_loss = 0
            num_answers = len(answers)
            for i in range(len(questions)):
                true_answer_idx = answers[i]
                true_answer = torch.zeros(num_answers).to(device)
                true_answer[true_answer_idx] = 1.0
                hard_loss += kl_divergence_loss(student_probs[i], true_answer)
            loss = 0.5 * soft_loss + 0.5 * hard_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

device = torch.device('cuda')

teacher_model.to(device)
student_model.to(device)

dataset = QA_Dataset(questions, answers, student_tokenizer)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = AdamW(student_model.parameters(), lr=2e-5)

train_model(train_loader, teacher_model, student_model, optimizer, device)

student_model.save_pretrained('COMPNet')
student_tokenizer.save_pretrained('COMPNet')