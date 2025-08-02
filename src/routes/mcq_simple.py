
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from youtube_transcript_api import YouTubeTranscriptApiimport 
import json
import tempfile
import openai
import google.generativeai as genai
from anthropic import Anthropic
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs
import pandas as pd
from io import BytesIO
import os
# import openai
# import google.generativeai as genai
# from anthropic import Anthropic

# Simplified version for deployment - PDF processing removed due to dependency conflicts
mcq_bp = Blueprint('mcq', __name__)

# Configuration for LLM APIs
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# Initialize clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if CLAUDE_API_KEY:
    anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        if parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
    return None

def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        raise Exception(f"Failed to get YouTube transcript: {str(e)}")

def chunk_text(text, max_tokens=3000):
    """Split text into chunks to respect LLM token limits"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + 1 > max_tokens * 0.75:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = 1
        else:
            current_chunk.append(word)
            current_length += 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def create_mcq_prompt(text, num_questions, difficulty):
    """Create prompt for MCQ generation"""
    bloom_levels = {
        "easy": "Remember and Understand (Bloom's Taxonomy levels 1-2): Focus on recall of facts, basic comprehension, and simple explanations.",
        "medium": "Apply and Analyze (Bloom's Taxonomy levels 3-4): Focus on application of concepts, analysis of relationships, and problem-solving.",
        "hard": "Evaluate and Create (Bloom's Taxonomy levels 5-6): Focus on critical evaluation, synthesis of ideas, and creation of new concepts."
    }
    
    bloom_instruction = bloom_levels.get(difficulty, bloom_levels["medium"])
    
    prompt = f"""Generate {num_questions} multiple-choice questions based on the following text. 

Difficulty Level: {difficulty.upper()}
{bloom_instruction}

Requirements:
1. Each question should have exactly 4 options (A, B, C, D)
2. Only one option should be correct
3. The other 3 options should be plausible but incorrect distractors
4. Include a brief explanation for the correct answer
5. Classify each question according to Bloom's taxonomy level
6. Questions should be clear, unambiguous, and directly related to the content

Text to analyze:
{text}

Please format your response as a JSON array with the following structure:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "Option A text",
      "B": "Option B text", 
      "C": "Option C text",
      "D": "Option D text"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation of why this is correct",
    "bloom_level": "Remember/Understand/Apply/Analyze/Evaluate/Create",
    "difficulty": "{difficulty}"
  }}
]

Ensure the JSON is valid and properly formatted."""
    
    return prompt

def generate_mcqs_with_llm(text, num_questions, difficulty, llm_provider="openai"):
    """Generate MCQs using specified LLM provider"""
    prompt = create_mcq_prompt(text, num_questions, difficulty)
    
    try:
        if llm_provider == "openai" and OPENAI_API_KEY:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educator who creates high-quality multiple-choice questions. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif llm_provider == "gemini" and GEMINI_API_KEY:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
            
        elif llm_provider == "claude" and CLAUDE_API_KEY:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        else:
            raise Exception(f"LLM provider {llm_provider} not configured or API key missing")
            
    except Exception as e:
        raise Exception(f"Failed to generate MCQs with {llm_provider}: {str(e)}")

@mcq_bp.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF file upload - simplified for deployment"""
    return jsonify({
        'error': 'PDF processing temporarily unavailable in deployed version. Please use the local version for PDF functionality.'
    }), 501

@mcq_bp.route('/extract-youtube', methods=['POST'])
def extract_youtube():
    """Extract transcript from YouTube video"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        video_id = extract_youtube_id(data['url'])
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        transcript = get_youtube_transcript(video_id)
        
        if not transcript:
            return jsonify({'error': 'No transcript available for this video'}), 400
        
        return jsonify({
            'success': True,
            'text': transcript,
            'video_id': video_id,
            'word_count': len(transcript.split())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcq_bp.route('/generate-mcqs', methods=['POST'])
def generate_mcqs():
    """Generate MCQs from provided text"""
    try:
        data = request.get_json()
        
        required_fields = ['text', 'num_questions', 'difficulty']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        text = data['text']
        num_questions = int(data['num_questions'])
        difficulty = data['difficulty'].lower()
        llm_provider = data.get('llm_provider', 'openai').lower()
        
        if difficulty not in ['easy', 'medium', 'hard']:
            return jsonify({'error': 'Difficulty must be easy, medium, or hard'}), 400
        
        if num_questions < 1 or num_questions > 50:
            return jsonify({'error': 'Number of questions must be between 1 and 50'}), 400
        
        chunks = chunk_text(text)
        all_mcqs = []
        
        questions_per_chunk = max(1, num_questions // len(chunks))
        remaining_questions = num_questions
        
        for i, chunk in enumerate(chunks):
            if remaining_questions <= 0:
                break
                
            chunk_questions = min(questions_per_chunk, remaining_questions)
            if i == len(chunks) - 1:
                chunk_questions = remaining_questions
            
            response = generate_mcqs_with_llm(chunk, chunk_questions, difficulty, llm_provider)
            
            try:
                chunk_mcqs = json.loads(response)
                if isinstance(chunk_mcqs, list):
                    all_mcqs.extend(chunk_mcqs)
                    remaining_questions -= len(chunk_mcqs)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    chunk_mcqs = json.loads(json_match.group())
                    if isinstance(chunk_mcqs, list):
                        all_mcqs.extend(chunk_mcqs)
                        remaining_questions -= len(chunk_mcqs)
        
        if not all_mcqs:
            return jsonify({'error': 'Failed to generate MCQs'}), 500
        
        all_mcqs = all_mcqs[:num_questions]
        
        return jsonify({
            'success': True,
            'mcqs': all_mcqs,
            'total_questions': len(all_mcqs)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcq_bp.route('/export-mcqs', methods=['POST'])
def export_mcqs():
    """Export MCQs in CSV format (simplified for deployment)"""
    try:
        data = request.get_json()
        
        if 'mcqs' not in data:
            return jsonify({'error': 'MCQs are required'}), 400
        
        mcqs = data['mcqs']
        
        # Simple CSV export
        csv_data = []
        for i, mcq in enumerate(mcqs, 1):
            row = {
                'Question_Number': i,
                'Question': mcq['question'],
                'Option_A': mcq['options']['A'],
                'Option_B': mcq['options']['B'],
                'Option_C': mcq['options']['C'],
                'Option_D': mcq['options']['D'],
                'Correct_Answer': mcq['correct_answer'],
                'Explanation': mcq.get('explanation', ''),
                'Bloom_Level': mcq.get('bloom_level', ''),
                'Difficulty': mcq.get('difficulty', '')
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='mcqs.csv',
            mimetype='text/csv'
        )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcq_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    available_providers = []
    if OPENAI_API_KEY:
        available_providers.append('openai')
    if GEMINI_API_KEY:
        available_providers.append('gemini')
    if CLAUDE_API_KEY:
        available_providers.append('claude')
    
    return jsonify({
        'status': 'healthy',
        'available_llm_providers': available_providers,
        'note': 'PDF processing unavailable in deployed version - use local version for full functionality'
    })

