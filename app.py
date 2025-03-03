from flask import Flask, request, jsonify
import google.generativeai as genai
from PIL import Image
from flask_cors import CORS
import io
import logging
import time
import os
import json
import psycopg2
from psycopg2.extras import Json
import base64
from datetime import datetime
from typing import Dict, Any

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS properly for Postman
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Database configuration - modify these values directly
DB_CONFIG = {
    'dbname': 'shop_data',
    'user': 'soubhikghosh',
    'password': 'hdfc@123',  # Replace with your actual password
    'host': 'localhost',
    'port': '5432'
}

# Configure Google API
GOOGLE_API_KEY = "AIzaSyCD6DGeERwWQbBC6BK1Hq0ecagQj72rqyQ"
genai.configure(api_key=GOOGLE_API_KEY)

# Other configuration
AUDIT_FOLDER = "audit"
PORT = 5001
DEBUG = True

@app.after_request
def after_request(response):
    """Add CORS headers after each request"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.before_request
def log_request():
    """Log basic request info"""
    logger.info(f"Request received: {request.method} {request.path}")

@app.after_request
def log_response(response):
    """Log basic response info"""
    logger.info(f"Response status: {response.status_code}")
    return response

def get_db_connection():
    """Create the database if it doesn't exist and return a connection."""
    try:
        # First try to connect to the default postgres database to check if our database exists
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        # Create database if it doesn't exist
        if not exists:
            logger.info(f"Database '{DB_CONFIG['dbname']}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            logger.info(f"Database '{DB_CONFIG['dbname']}' created successfully")
        
        cursor.close()
        conn.close()
        
        # Now connect to our actual database
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"Connected to database '{DB_CONFIG['dbname']}'")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """Initialize database tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create shops table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shops (
            id SERIAL PRIMARY KEY,
            location_data JSONB,
            shop_inference JSONB,
            image_data BYTEA,
            created_at TIMESTAMP
        )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_shops_created_at ON shops(created_at);
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def store_shop_data(location_data, shop_inference, image_data):
    """Store shop data in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO shops (location_data, shop_inference, image_data, created_at)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        ''', (
            Json(location_data),
            Json(shop_inference),
            image_data,
            datetime.now()
        ))
        
        shop_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Shop data stored successfully with ID: {shop_id}")
        return shop_id
    except Exception as e:
        logger.error(f"Error storing shop data: {str(e)}")
        raise

def create_analysis_prompt(image_type: str = "shop") -> str:
    """Creates a robust prompt for multilingual image analysis."""
    return f"""
    Analyze this image with extreme attention to detail and skepticism. First verify if this is a legitimate {image_type} image.
    
    For ALL text you find in the image:
    1. Identify the original language
    2. Provide both the original text and its English translation if not in English
    3. Note the location/context of where this text appears in the image
    4. Look for text in ALL parts of the image, including small print, background signs, and partially visible text
    
    If it's not a legitimate {image_type} image, respond only with the basic JSON structure marking is_valid as false.
    
    If it is legitimate, carefully extract the following information:
    1. All visible text as specified above
    2. The physical characteristics and objects visible
    3. The overall setting and context
    4. Any cultural indicators that might help identify the region/locality
    
    Provide your analysis in the following strict JSON structure only, with no additional text:
    {{
        "is_valid": true/false,
        "shop_details": {{
            "name": {{
                "original_text": "text as written",
                "language": "detected language",
                "english_translation": "translation if needed, same as original if English",
                "confidence": "high/medium/low"
            }},
            "location": {{
                "original_text": "text as written",
                "language": "detected language",
                "english_translation": "translation if needed, same as original if English",
                "detected_country_or_region": "based on visual cues and text"
            }},
            "additional_text": [
                {{
                    "original_text": "text as written",
                    "language": "detected language",
                    "english_translation": "translation if needed",
                    "context": "where this text appears in image"
                }}
            ]
        }},
        "physical_analysis": {{
            "visible_objects": ["list of main objects visible"],
            "setting_description": "brief setting description",
            "cultural_indicators": ["list of cultural/regional indicators observed"]
        }},
        "business_inference": {{
            "primary_business_type": "inferred business type",
            "confidence_score": "high/medium/low",
            "reasoning": "brief explanation",
            "likely_target_market": "inferred target demographic"
        }},
        "analysis_metadata": {{
            "analysis_timestamp": "timestamp",
            "is_shop": true/false,
            "languages_detected": ["list of all languages found in image"]
        }}
    }}
    
    Remember: 
    1. Return ONLY valid JSON, no other text or explanation
    2. Always provide both original text and English translations when non-English text is found
    3. Include confidence levels for text recognition
    4. Note any signs of text manipulation or digital alteration
    """

def analyze_image(image_data: bytes) -> Dict[str, Any]:
    """Analyzes image using Gemini API with error handling and validation."""
    try:
        start_time = time.time()
        logger.info("Starting image analysis")
        
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Open image
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"Image opened successfully")
        
        # Get analysis
        prompt = create_analysis_prompt()
        response = model.generate_content([prompt, image])
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

        json_str = response.text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
            
        resp = json.loads(json_str.strip())
        
        return resp
        
    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        return {
            "is_valid": False,
            "error": str(e),
            "analysis_metadata": {
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_shop": False,
                "languages_detected": []
            }
        }

@app.route('/analyze-shop', methods=['POST', 'OPTIONS'])
def analyze_shop():
    """Endpoint to analyze shop images."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({
                "error": "No image provided",
                "status": "error"
            }), 400

        image_file = request.files['image']
        logger.info(f"Processing image: {image_file.filename}")
        
        image_data = image_file.read()

        if not image_data:
            logger.warning("Empty image data received")
            return jsonify({
                "error": "Empty image data",
                "status": "error"
            }), 400

        analysis_result = analyze_image(image_data)
        
        return jsonify(analysis_result), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "error"
        }), 500

@app.route('/submit-shop', methods=['POST', 'OPTIONS'])
def submit_shop():
    """Endpoint to accept shop location, inference, and image."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Check if image is provided
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({
                "error": "No image provided",
                "status": "error"
            }), 400
        
        # Get image data
        image_file = request.files['image']
        image_data = image_file.read()
        
        if not image_data:
            logger.warning("Empty image data received")
            return jsonify({
                "error": "Empty image data",
                "status": "error"
            }), 400
        
        # Get JSON data
        if not request.form.get('shop_data'):
            logger.warning("No shop data in request")
            return jsonify({
                "error": "No shop data provided",
                "status": "error"
            }), 400
            
        shop_data = json.loads(request.form.get('shop_data'))
        
        # Extract required fields
        location_data = shop_data.get('location', {})
        shop_inference = shop_data.get('inference', {})
        
        # Save image to audit folder
        if not os.path.exists(AUDIT_FOLDER):
            os.makedirs(AUDIT_FOLDER)
            
        timestamp = time.strftime("%Y%m%d%H%M%S")
        image_path = os.path.join(AUDIT_FOLDER, f"{timestamp}.jpg")
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        logger.info(f"Saved audit image to {image_path}")
        
        # Store data in database
        shop_id = store_shop_data(location_data, shop_inference, image_data)
        
        return jsonify({
            "status": "success",
            "shop_id": shop_id,
            "message": "Shop data successfully stored"
        }), 200
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON data provided")
        return jsonify({
            "error": "Invalid JSON data",
            "status": "error"
        }), 400
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "service": "shop-analyzer"
    }), 200

if __name__ == '__main__':
    logger.info("Starting shop analyzer service")
    # Initialize database - this will create the DB if it doesn't exist
    init_db()
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)