from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from models import update_knowledge_base, update_instruction
from datetime import datetime

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['chat_database']
sessions_collection = db['sessions']
sessions_collection.create_index('session_id')  # Create index for better query performance
instructions_collection = db['instructions']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the latest instruction from MongoDB or set default

def load_latest_instruction():
    latest_instruction = instructions_collection.find_one(sort=[("timestamp", -1)])
    if latest_instruction:
        return latest_instruction['instruction']
    return "You are a knowledgeable and friendly assistant for TechBerry."

current_instruction = load_latest_instruction()

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new session and return a session ID."""
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'Invalid payload or missing user_id.'}), 400

    user_id = data['user_id'].strip()
    if not user_id:
        return jsonify({'error': 'User ID is required.'}), 400

    # Get the latest instruction ID
    latest_instruction = instructions_collection.find_one(sort=[("timestamp", -1)])
    if not latest_instruction:
        return jsonify({'error': 'No instructions found in the database.'}), 500

    session = {
        'user_id': user_id,
        'session_id': str(ObjectId()),
        'current_instruction_id': str(latest_instruction['_id']),
        'messages': []
    }
    sessions_collection.insert_one(session)
    return jsonify({'session_id': session['session_id']}), 200

@app.route('/save_message', methods=['POST'])
def save_message():
    """Save a user message and bot response in the database."""
    data = request.get_json()
    if not data or not all(k in data for k in ['session_id', 'message', 'response']):
        return jsonify({'error': 'Invalid payload or missing fields.'}), 400

    session_id = data['session_id'].strip()
    user_message = data['message'].strip()
    bot_response = data['response'].strip()

    try:
        session = sessions_collection.find_one({'session_id': session_id})
        if not session:
            return jsonify({'error': 'Invalid session ID.'}), 404

        sessions_collection.update_one(
            {'session_id': session_id},
            {'$push': {'messages': {'user_message': user_message, 'bot_response': bot_response}}}
        )
        return jsonify({'message': 'Message saved successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f'MongoDB error: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Process a user question, generate an answer, and save the conversation in the session."""
    data = request.get_json()
    if not data or not all(k in data for k in ['session_id', 'question']):
        return jsonify({'error': 'Invalid payload or missing fields.'}), 400

    session_id = data['session_id'].strip()
    question = data['question'].strip()

    try:
        # Retrieve the session's conversation history
        session = sessions_collection.find_one({'session_id': session_id})
        if not session:
            return jsonify({'error': 'Invalid session ID.'}), 404

        # Retrieve the instruction associated with the session
        instruction_id = session['current_instruction_id']
        instruction = instructions_collection.find_one({'_id': ObjectId(instruction_id)})
        if not instruction:
            return jsonify({'error': 'Instruction not found for this session.'}), 404

        # Build context from session messages
        context = "\n".join(
            [f"User: {msg['user_message']}\nBot: {msg['bot_response']}" for msg in session['messages']]
        )

        # Generate response using the instruction and context
        from models import rag_chain
        prompt = f"{instruction['instruction']}\nContext: {context}\nQuestion: {question}\nAnswer:"
        answer = rag_chain(prompt)

        # Handle empty or invalid responses
        if not answer or answer.strip() == "":
            answer = "I'm sorry, I cannot find this information. Please try asking differently."

        # Save the new message to the session
        sessions_collection.update_one(
            {'session_id': session_id},
            {'$push': {'messages': {'user_message': question, 'bot_response': answer}}}
        )
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'MongoDB error: {str(e)}'}), 500


@app.route('/upload_knowledge_base', methods=['POST'])
def upload_knowledge_base():
    """Upload a file and update the knowledge base and vector store."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Save the file to the 'data' folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Update the knowledge base and vector store
        from models import update_knowledge_base  # Import the function
        update_knowledge_base(file_path)  # Pass the file to be processed and stored

        return jsonify({'message': f'File {filename} successfully uploaded and added to the vector store.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to process the file: {str(e)}'}), 500

@app.route('/set_instruction', methods=['POST'])
def set_instruction():
    """Update the instruction prompt used by the model."""
    data = request.get_json()
    if not data or 'instruction' not in data:
        return jsonify({'error': 'Instruction field is required.'}), 400

    instruction = data['instruction'].strip()
    if not instruction:
        return jsonify({'error': 'Instruction cannot be empty.'}), 400

    try:
        instructions_collection.insert_one({
            'instruction': instruction,
            'timestamp': datetime.utcnow()
        })
        return jsonify({'message': 'Instruction updated successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to update instruction: {str(e)}'}), 500

@app.route('/get_instruction', methods=['POST'])
def get_instruction():
    """Retrieve the instruction for a given user ID."""
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'User ID is required.'}), 400

    user_id = data['user_id'].strip()
    if not user_id:
        return jsonify({'error': 'User ID cannot be empty.'}), 400

    try:
        session = sessions_collection.find_one({'user_id': user_id})
        if not session:
            return jsonify({'error': 'Session not found for this user.'}), 404

        instruction_id = session['current_instruction_id']
        instruction = instructions_collection.find_one({'_id': ObjectId(instruction_id)})
        if not instruction:
            return jsonify({'error': 'Instruction not found for this session.'}), 404

        return jsonify({'instruction': instruction['instruction']}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve instruction: {str(e)}'}), 500

def create_session_with_instruction(user_id: str) -> str:
    # ดึง Instruction ล่าสุดจาก MongoDB
    latest_instruction = db['instructions'].find_one(sort=[("timestamp", -1)])
    if not latest_instruction:
        raise ValueError("No instructions found in the database.")

    session = {
        'user_id': user_id,
        'session_id': str(ObjectId()),
        'current_instruction_id': str(latest_instruction["_id"]),
        'messages': []
    }
    sessions_collection.insert_one(session)
    return session['session_id']

@app.route('/create_session', methods=['POST'])
def create_session():
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'User ID is required.'}), 400

    user_id = data['user_id'].strip()
    if not user_id:
        return jsonify({'error': 'User ID cannot be empty.'}), 400

    try:
        session_id = create_session_with_instruction(user_id)
        return jsonify({'message': 'Session created successfully.', 'session_id': session_id}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to create session: {str(e)}'}), 500

@app.route('/add_instruction', methods=['POST'])
def add_instruction():
    """Add a new instruction to the database without updating the runtime prompt."""
    data = request.get_json()
    if not data or 'instruction' not in data:
        return jsonify({'error': 'Instruction field is required.'}), 400

    instruction = data['instruction'].strip()
    if not instruction:
        return jsonify({'error': 'Instruction cannot be empty.'}), 400

    try:
        # เพิ่ม Instruction ลง MongoDB
        instruction_id = db['instructions'].insert_one({
            "instruction": instruction,
            "timestamp": datetime.utcnow()
        }).inserted_id
        return jsonify({'message': 'Instruction added successfully.', 'instruction_id': str(instruction_id)}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to add instruction: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
