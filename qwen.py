from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import spacy
import string
import re
import collections
import torch

# Load Spacy model for semantic similarity
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model...")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Initialize Tokenizer and Model
model_name = "MiniLLM/MiniPLM-Qwen-1.2B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto")

def get_answer(prompt):
    # ChatML format or simple completion? The user asked for "Direct Ask".
    # Since it's a base model (or fine-tuned?), usually CausalLMs need clear stop tokens.
    # We will use the prompt as is.
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, # Keep it short for answers
            do_sample=False,   # Greedy decoding for reproducibility
            repetition_penalty=1.2, # Prevent repetition loops
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice off the input prompt to get only the generated answer
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # If the model repeats the prompt, we strip it.
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        answer = generated_text.strip()
        
    # Heuristic: Cut off at specific tokens to stop hallucinated continuations
    stop_markers = ["Question:", "Text:", "\n"]
    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()
        
    return answer

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def semantic_similarity(prediction, ground_truth):
    doc1 = nlp(prediction)
    doc2 = nlp(ground_truth)
    if not doc1.vector_norm or not doc2.vector_norm:
        return 0.0
    return doc1.similarity(doc2)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def process_questions(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found.")
        return

    print("Running Zero-Shot Prompting (The 'Direct Ask') with MiniPLM-Qwen-1.2B...")

    total_em = 0
    total_f1 = 0
    total_similarity = 0
    count = 0

    for i, item in enumerate(questions):
        context = item['context']
        question = item['question']
        ground_truths = [ans['text'] for ans in item['answers']]
        
        # Zero-Shot Prompt
        prompt = f"Read the following text and answer the question. Text: {context} Question: {question} Answer:"
        
        prediction = get_answer(prompt)
        
        em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        similarity = metric_max_over_ground_truths(semantic_similarity, prediction, ground_truths)
        
        total_em += em
        total_f1 += f1
        total_similarity += similarity
        count += 1
        
        print(f"Question {i+1}: {question}")
        print(f"Prediction: {prediction}")
        print(f"Answers: {ground_truths}")
        print(f"EM: {em}, F1: {f1:.2f}, Similarity: {similarity:.2f}")
        print("-" * 50)
    
    if count > 0:
        print(f"\nEvaluation Results over {count} questions (Zero-Shot):")
        print(f"Average Exact Match: {100.0 * total_em / count:.2f}")
        print(f"Average F1 Score: {100.0 * total_f1 / count:.2f}")
        print(f"Average Semantic Similarity: {total_similarity / count:.2f}")

if __name__ == "__main__":
    process_questions("filtered_questions.json")
