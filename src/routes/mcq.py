import os
import json
import tempfile
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None
import pdfplumber
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import openai
import google.generativeai as genai
from anthropic import Anthropic
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    try:
        from mistralai import Mistral
        MistralClient = Mistral
        ChatMessage = None
    except ImportError:
        MistralClient = None
        ChatMessage = None

mcq_bp = Blueprint('mcq', __name__)

# Configuration for LLM APIs (these should be set via environment variables)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Initialize clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if CLAUDE_API_KEY:
    anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

if MISTRAL_API_KEY and MistralClient:
    try:
        mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
    except Exception:
        mistral_client = None
else:
    mistral_client = None

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

def extract_pdf_text(file_path):
    """Extract text from PDF using PyMuPDF or pdfplumber as fallback"""
    text = ""
    
    # Try PyMuPDF first if available
    if PYMUPDF_AVAILABLE and fitz:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"PyMuPDF failed: {e}")
    
    # Fallback to pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to extract PDF text: {str(e)}")

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
        # Rough estimation: 1 token ≈ 0.75 words
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
            model = genai.GenerativeModel('gemini-2.5-flash')
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
            
        elif llm_provider == "mistral" and MISTRAL_API_KEY and mistral_client:
            try:
                if ChatMessage:
                    response = mistral_client.chat(
                        model="mistral-large-latest",
                        messages=[ChatMessage(role="user", content=prompt)]
                    )
                    return response.choices[0].message.content
                else:
                    # Fallback for different API structure
                    response = mistral_client.chat(
                        model="mistral-large-latest",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content
            except Exception as e:
                raise Exception(f"Mistral API error: {str(e)}")
            
        else:
            raise Exception(f"LLM provider {llm_provider} not configured or API key missing")
            
    except Exception as e:
        raise Exception(f"Failed to generate MCQs with {llm_provider}: {str(e)}")

@mcq_bp.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF file upload and text extraction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Extract text from PDF
        text = extract_pdf_text(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if not text:
            return jsonify({'error': 'No text could be extracted from the PDF'}), 400
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': filename,
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        # Validate input
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
        
        # Handle long text by chunking
        chunks = chunk_text(text)
        all_mcqs = []
        
        questions_per_chunk = max(1, num_questions // len(chunks))
        remaining_questions = num_questions
        
        for i, chunk in enumerate(chunks):
            if remaining_questions <= 0:
                break
                
            chunk_questions = min(questions_per_chunk, remaining_questions)
            if i == len(chunks) - 1:  # Last chunk gets remaining questions
                chunk_questions = remaining_questions
            
            response = generate_mcqs_with_llm(chunk, chunk_questions, difficulty, llm_provider)
            
            # Parse JSON response
            try:
                chunk_mcqs = json.loads(response)
                if isinstance(chunk_mcqs, list):
                    all_mcqs.extend(chunk_mcqs)
                    remaining_questions -= len(chunk_mcqs)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    chunk_mcqs = json.loads(json_match.group())
                    if isinstance(chunk_mcqs, list):
                        all_mcqs.extend(chunk_mcqs)
                        remaining_questions -= len(chunk_mcqs)
        
        if not all_mcqs:
            return jsonify({'error': 'Failed to generate MCQs'}), 500
        
        # Limit to requested number of questions
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
    """Export MCQs in various formats"""
    try:
        data = request.get_json()
        
        if 'mcqs' not in data or 'format' not in data:
            return jsonify({'error': 'MCQs and format are required'}), 400
        
        mcqs = data['mcqs']
        export_format = data['format'].lower()
        
        if export_format == 'pdf':
            return export_pdf(mcqs)
        elif export_format == 'csv':
            return export_csv(mcqs)
        elif export_format == 'html':
            return export_html(mcqs)
        else:
            return jsonify({'error': 'Unsupported format. Use pdf, csv, or html'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def export_pdf(mcqs):
    """Export MCQs as PDF"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Multiple Choice Questions", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    for i, mcq in enumerate(mcqs, 1):
        # Question
        question_text = f"<b>Question {i}:</b> {mcq['question']}"
        question_para = Paragraph(question_text, styles['Normal'])
        story.append(question_para)
        story.append(Spacer(1, 6))
        
        # Options
        for option_key, option_text in mcq['options'].items():
            option_para = Paragraph(f"{option_key}. {option_text}", styles['Normal'])
            story.append(option_para)
        
        # Correct answer and explanation
        answer_text = f"<b>Correct Answer:</b> {mcq['correct_answer']}"
        answer_para = Paragraph(answer_text, styles['Normal'])
        story.append(answer_para)
        
        if 'explanation' in mcq:
            explanation_text = f"<b>Explanation:</b> {mcq['explanation']}"
            explanation_para = Paragraph(explanation_text, styles['Normal'])
            story.append(explanation_para)
        
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name='mcqs.pdf',
        mimetype='application/pdf'
    )

def export_csv(mcqs):
    """Export MCQs as CSV"""
    data = []
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
        data.append(row)
    
    df = pd.DataFrame(data)
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name='mcqs.csv',
        mimetype='text/csv'
    )

def export_html(mcqs):
    """Export MCQs as HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multiple Choice Questions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .question { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .question-title { font-weight: bold; font-size: 16px; margin-bottom: 10px; }
            .options { margin: 10px 0; }
            .option { margin: 5px 0; }
            .answer { background-color: #e8f5e8; padding: 10px; margin-top: 10px; border-radius: 3px; }
            .explanation { font-style: italic; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>Multiple Choice Questions</h1>
    """
    
    for i, mcq in enumerate(mcqs, 1):
        html_content += f"""
        <div class="question">
            <div class="question-title">Question {i}: {mcq['question']}</div>
            <div class="options">
        """
        
        for option_key, option_text in mcq['options'].items():
            html_content += f'<div class="option">{option_key}. {option_text}</div>'
        
        html_content += f"""
            </div>
            <div class="answer">
                <strong>Correct Answer:</strong> {mcq['correct_answer']}
        """
        
        if 'explanation' in mcq:
            html_content += f'<div class="explanation"><strong>Explanation:</strong> {mcq["explanation"]}</div>'
        
        html_content += "</div></div>"
    
    html_content += "</body></html>"
    
    buffer = BytesIO()
    buffer.write(html_content.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name='mcqs.html',
        mimetype='text/html'
    )

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
    if MISTRAL_API_KEY:
        available_providers.append('mistral')
    
    return jsonify({
        'status': 'healthy',
        'available_llm_providers': available_providers
    })

