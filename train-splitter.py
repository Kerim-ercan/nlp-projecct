import json

def filter_questions(input_file, output_file, target_count=150):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []
    count = 0
    keywords = ("Who", "What", "Where", "When")

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question'].strip()
                if question.startswith(keywords):
                    filtered_data.append({
                        "context": context,
                        "question": question,
                        "id": qa['id'],
                        "answers": qa['answers']
                    })
                    count += 1
                    
                    if count >= target_count:
                        break
            if count >= target_count:
                break
        if count >= target_count:
            break

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Successfully extracted {len(filtered_data)} questions to {output_file}")

if __name__ == "__main__":
    filter_questions('train-v2.0.json', 'filtered_questions.json')
