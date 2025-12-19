#https://huggingface.co/ozcangundes/mt5-small-turkish-squad
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import spacy
import string
import re
import collections

# Load Spacy model for semantic similarity
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model...")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-small-turkish-squad")
model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-small-turkish-squad")

FEW_SHOT_EXAMPLES = """Text: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. Question: Who gave their name to Normandy? Answer: The Normans
Text: The Sun is the star at the center of the Solar System. It is a nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core. Question: What is at the center of the Solar System? Answer: The Sun
Text: Beyoncé Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child. Question: Where was Beyonce born? Answer: Houston, Texas
"""

def get_answer(prompt):
  source_encoding=tokenizer(
    prompt,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt")
  
  generated_ids=model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      max_length=120)

  preds=[tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in generated_ids]

  return "".join(preds)

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

    print("Select prompting mode:")
    print("1. Zero-Shot")
    print("2. Few-Shot")
    print("3. Chain-of-Thought")
    
    while True:
        mode = input("Enter number (1, 2, or 3): ").strip()
        if mode in ['1', '2', '3']:
            break
        print("Invalid selection. Please enter 1, 2, or 3.")

    total_em = 0
    total_f1 = 0
    total_similarity = 0
    count = 0

    mode_names = {'1': 'Zero-Shot', '2': 'Few-Shot', '3': 'Chain-of-Thought'}
    print(f"\nRunning in {mode_names[mode]} mode...\n")

    for i, item in enumerate(questions):
        context = item['context']
        question = item['question']
        ground_truths = [ans['text'] for ans in item['answers']]
        
        if mode == '2': # Few-Shot
            prompt = f"{FEW_SHOT_EXAMPLES}Read the following text and answer the question. Text: {context} Question: {question} Answer:"
        elif mode == '3': # Chain-of-Thought
            prompt = f"Read the text and answer the question. Explain your reasoning step-by-step before giving the final answer. Text: {context} Question: {question} Answer:"
        else: # Zero-Shot
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
        print(f"\nEvaluation Results over {count} questions ({mode_names[mode]}):")
        print(f"Average Exact Match: {100.0 * total_em / count:.2f}")
        print(f"Average F1 Score: {100.0 * total_f1 / count:.2f}")
        print(f"Average Semantic Similarity: {total_similarity / count:.2f}")

if __name__ == "__main__":
    process_questions("filtered_questions.json")
