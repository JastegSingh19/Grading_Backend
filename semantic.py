import zipfile
import os
import fitz  # PyMuPDF
import json
import pandas as pd
import spacy
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Load pre-trained model tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def extract_keywords_and_entities(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    entities = [ent.text.lower() for ent in doc.ents]
    return keywords, entities


def encode_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def calculate_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)[0][0]


def keyword_matching(answer, keywords, threshold=0.8):
    answer_lower = answer.lower()
    match_count = sum(keyword.lower() in answer_lower for keyword in keywords)
    return match_count / len(keywords) >= threshold


def named_entity_recognition(answer, expected_entities, threshold=0.5):
    doc = nlp(answer.lower())
    entities = [ent.text.lower() for ent in doc.ents]
    match_count = sum(any(entity in answer_entity for answer_entity in entities) for entity in expected_entities)
    return match_count / len(expected_entities) >= threshold


def extract_pdfs_from_zip(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def convert_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text


def split_text_by_questions(text):
    questions = text.split('\n\n')
    return [q.strip() for q in questions if q.strip()]


def assign_grade(similarity, keywords_match, entities_match):
    if keywords_match and entities_match:
        return 'A+'
    elif keywords_match and not entities_match:
        return 'C+'
    elif entities_match and not keywords_match:
        return 'B+'
    else:
        return 'F'


def grade_individual_questions(correct_answers, student_answers):
    results = []

    for student_id, answers in student_answers.items():
        student_results = {'student_id': student_id}
        for i, (correct_answer, student_answer) in enumerate(zip(correct_answers, answers)):
            question_id = f'Q{i + 1}'
            keywords, expected_entities = extract_keywords_and_entities(correct_answer)
            correct_vector = encode_sentence(correct_answer, tokenizer, model)
            student_vector = encode_sentence(student_answer, tokenizer, model)

            similarity = float(calculate_similarity(correct_vector, student_vector))
            keywords_match = keyword_matching(student_answer, keywords)
            entities_match = named_entity_recognition(student_answer, expected_entities)
            grade = assign_grade(similarity, keywords_match, entities_match)

            student_results[f'{question_id}_similarity'] = similarity
            student_results[f'{question_id}_keywords_match'] = keywords_match
            student_results[f'{question_id}_entities_match'] = entities_match
            student_results[f'{question_id}_grade'] = grade

        results.append(student_results)

    return results


def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


@app.route('/grade', methods=['POST'])
def grade():
    if 'zip_path' not in request.files or 'correct_answer_pdf' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    zip_file = request.files['zip_path']
    correct_answer_file = request.files['correct_answer_pdf']

    zip_path = 'temp_zip.zip'
    correct_answer_pdf_path = 'temp_correct_answer.pdf'

    zip_file.save(zip_path)
    correct_answer_file.save(correct_answer_pdf_path)

    extract_to = 'extracted_pdfs'
    output_csv = 'grading_results.csv'

    # Extract PDFs from the ZIP file
    extract_pdfs_from_zip(zip_path, extract_to)

    # Convert the correct answer PDF to text
    correct_answer_text = convert_pdf_to_text(correct_answer_pdf_path)

    # Split the correct answer text into individual questions
    correct_answers = split_text_by_questions(correct_answer_text)

    # Process each student's answers one by one
    results = []
    for filename in os.listdir(extract_to):
        if filename.endswith('.pdf') and filename != os.path.basename(correct_answer_pdf_path):
            pdf_path = os.path.join(extract_to, filename)
            student_text = convert_pdf_to_text(pdf_path)
            student_answers = split_text_by_questions(student_text)
            student_answers_dict = {filename: student_answers}

            # Grade the student answers for each question
            result = grade_individual_questions(correct_answers, student_answers_dict)
            results.extend(result)

    # Save the results to a CSV file
    save_results_to_csv(results, output_csv)

    with open(output_csv, 'r') as f:
        csv_content = f.read()

    return csv_content, 200, {'Content-Type': 'text/csv'}


'''if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')'''
