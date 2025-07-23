#!/usr/bin/env python3
"""
Emoji Search API - Flask Web API for emoji vector search

RESTful API interface for finding similar emojis using vector similarity and emotion-based search.
"""

import os
import json
import logging
import base64
import uuid
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError
from werkzeug.utils import secure_filename
from prompts import EMOJI_ANALYSIS_PROMPT, EMOJI_ANALYSIS_SYSTEM_MESSAGE, TEXT_ANALYSIS_PROMPT_TEMPLATE, TEXT_ANALYSIS_SYSTEM_MESSAGE
from emoji_vector_storage import EmojiVectorStorage

# Import PIL for image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL (Pillow) not available. Image conversion disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Text analysis endpoint will be disabled.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'downloaded_emojis'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage instance
storage = None

class EmojiSearchAPI:
    """API wrapper for emoji vector search functionality"""
    
    def __init__(self, config_file: str = "azure_config.json"):
        """Initialize the API with storage backend"""
        self.storage = EmojiVectorStorage(config_file)
        self.config = self.storage.config
        self.logger = logger
        self.logger.info("🔍 Emoji Search API initialized")
        
        # Initialize Azure OpenAI client for text analysis
        self.openai_client = None
        if AZURE_OPENAI_AVAILABLE:
            try:
                self.openai_client = AzureOpenAI(
                    api_key=self.config.get('azure_openai_api_key'),
                    api_version=self.config.get('azure_openai_api_version', '2024-02-15-preview'),
                    azure_endpoint=self.config.get('azure_openai_endpoint')
                )
                self.deployment_name = self.config.get('deployment_name', 'gpt-4o')
                self.logger.info("✅ Azure OpenAI client initialized for text analysis")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Azure OpenAI client: {e}")
                self.openai_client = None
    
    def _convert_string_to_array(self, data) -> List[str]:
        """Convert comma-separated string back to array"""
        if isinstance(data, list):
            return data
        if isinstance(data, str) and data.strip():
            return [item.strip() for item in data.split(',') if item.strip()]
        return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the emoji database"""
        try:
            # Get total count
            results = self.storage.search_client.search(
                search_text="*",
                select=["id"],
                top=1000  # Assuming we don't have more than 1000 emojis
            )
            
            total_count = 0
            for result in results:
                total_count += 1
            
            return {
                'total_count': total_count,
                'timestamp': datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Stats retrieval failed: {e}")
            raise InternalServerError(f"Stats retrieval failed: {str(e)}")
    
    def search_by_name(self, emoji_name: str) -> Dict:
        """Search for a specific emoji by name and return its details"""
        try:
            if not emoji_name.strip():
                raise BadRequest("Emoji name cannot be empty")
            
            # Clean the emoji name (remove file extensions if present)
            clean_name = emoji_name.strip().lower()
            if clean_name.endswith('.png'):
                clean_name = clean_name[:-4]
            
            # Search for exact match first
            results = self.storage.search_client.search(
                search_text=f'emoji_name:"{clean_name}"',
                select=["id", "emoji_name", "usage_scenarios"],
                top=1
            )
            
            result_list = list(results)
            
            # If no exact match, try partial match
            if not result_list:
                results = self.storage.search_client.search(
                    search_text=clean_name,
                    select=["id", "emoji_name", "usage_scenarios"],
                    top=5  # Return up to 5 partial matches
                )
                result_list = list(results)
            
            if not result_list:
                return {
                    "results": [],
                    "emoji_name": emoji_name,
                    "count": 0,
                    "message": f"No emoji found with name '{emoji_name}'",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            
            # Format results
            formatted_results = []
            for result in result_list:
                # Convert usage_scenarios string back to array if needed
                usage_scenarios = self._convert_string_to_array(result.get('usage_scenarios', []))
                
                emoji_data = {
                    'id': result['id'],
                    'emoji_name': result['emoji_name'],
                    'usage_scenarios': usage_scenarios,
                    'score': result.get('@search.score', 0),
                    'image_path': f"/static/downloaded_emojis/{result['emoji_name']}.png"
                }
                formatted_results.append(emoji_data)
            
            return {
                "results": formatted_results,
                "emoji_name": emoji_name,
                "count": len(formatted_results),
                "match_type": "exact" if len(result_list) == 1 and result_list[0]['emoji_name'].lower() == clean_name else "partial",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
        except BadRequest:
            raise
        except Exception as e:
            self.logger.error(f"❌ Name search failed: {e}")
            raise InternalServerError(f"Name search failed: {str(e)}")
    
    def search_by_filename(self, filename: str, top_k: int = 10) -> Dict:
        """Search for emoji index by filename/emoji name containing the query string"""
        try:
            if not filename.strip():
                raise BadRequest("Filename cannot be empty")
            
            # Clean the query string - remove .png extension for more flexible searching
            clean_query = filename.strip().lower()
            if clean_query.endswith('.png'):
                clean_query = clean_query[:-4]
            
            # Since filename field is only filterable (not searchable), we'll use emoji_name field 
            # which is searchable and should match the filename (without .png extension)
            # First try searching in emoji_name field with full-text search
            results = self.storage.search_client.search(
                search_text=f"{clean_query}*",
                search_fields=["emoji_name"],
                select=["id", "emoji_name", "filename", "primary_emotion", "secondary_emotions", "usage_scenarios", "tone", "context_suggestions"],
                top=top_k
            )
            
            result_list = list(results)
            
            # If no results with prefix search, try broader search in emoji_name 
            if not result_list:
                results = self.storage.search_client.search(
                    search_text=clean_query,
                    search_fields=["emoji_name"],
                    select=["id", "emoji_name", "filename", "primary_emotion", "secondary_emotions", "usage_scenarios", "tone", "context_suggestions"],
                    top=top_k
                )
                result_list = list(results)
                match_type = "contains"
            else:
                match_type = "starts_with"
            
            if not result_list:
                return {
                    "results": [],
                    "filename": filename,
                    "query": clean_query,
                    "count": 0,
                    "message": f"No emoji index found with filename containing '{filename}'",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            
            # Format results
            formatted_results = []
            for result in result_list:
                # Convert string fields back to arrays if needed
                secondary_emotions = self._convert_string_to_array(result.get('secondary_emotions', []))
                usage_scenarios = self._convert_string_to_array(result.get('usage_scenarios', []))
                context_suggestions = self._convert_string_to_array(result.get('context_suggestions', []))
                
                emoji_data = {
                    'id': result['id'],
                    'emoji_name': result.get('emoji_name', ''),
                    'filename': result.get('filename', ''),
                    'primary_emotion': result.get('primary_emotion', ''),
                    'secondary_emotions': secondary_emotions,
                    'usage_scenarios': usage_scenarios,
                    'tone': result.get('tone', ''),
                    'context_suggestions': context_suggestions,
                    'score': result.get('@search.score', 1.0),
                    'image_path': f"/static/downloaded_emojis/{result.get('filename', '')}"
                }
                formatted_results.append(emoji_data)
            
            return {
                "results": formatted_results,
                "filename": filename,
                "query": clean_query,
                "count": len(formatted_results),
                "match_type": match_type,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
        except BadRequest:
            raise
        except Exception as e:
            self.logger.error(f"❌ Filename search failed: {e}")
            raise InternalServerError(f"Filename search failed: {str(e)}")
    
    def analyze_text_and_search(self, text: str, top_k: int = 5) -> Dict:
        """Analyze text using LLM to extract emotions and context, then search for matching emojis"""
        try:
            if not text.strip():
                raise BadRequest("Text cannot be empty")
            
            if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
                raise BadRequest("top_k must be an integer between 1 and 50")
            
            if not self.openai_client:
                raise InternalServerError("Text analysis service is not available")
            
            # Analyze text using Azure OpenAI
            analysis = self._analyze_text_with_llm(text)
            if not analysis:
                raise InternalServerError("Failed to analyze text with LLM")
            
            # Create search query from analysis results
            search_query = self._create_search_query_from_analysis(analysis)
            
            # Search for matching emojis using vector similarity
            emoji_results = self.storage.search_similar_emojis(search_query, top_k)
            
            # Ensure usage scenarios are included in results and add image paths
            final_results = []
            for result in emoji_results:
                # If usage_scenarios is missing, try to fetch it from the database
                if 'usage_scenarios' not in result and 'emoji_name' in result:
                    # Fetch full record to get usage scenarios
                    full_result = self.storage.search_client.search(
                        search_text=f'emoji_name:"{result["emoji_name"]}"',
                        select=["id", "emoji_name", "usage_scenarios"],
                        top=1
                    )
                    full_result_list = list(full_result)
                    if full_result_list:
                        result['usage_scenarios'] = full_result_list[0].get('usage_scenarios', [])
                
                # Convert usage_scenarios string back to array if needed
                if 'usage_scenarios' in result:
                    result['usage_scenarios'] = self._convert_string_to_array(result['usage_scenarios'])
                
                result['match_type'] = 'vector_similarity'
                # Ensure image path is included
                if 'emoji_name' in result and 'image_path' not in result:
                    result['image_path'] = f"/static/downloaded_emojis/{result['emoji_name']}.png"
                final_results.append(result)
            
            self.logger.info(f"🎯 Found {len(final_results)} emojis using vector similarity")
            
            return {
                "analysis": analysis,
                "search_query": search_query,
                "emoji_results": final_results,
                "text": text,
                "count": len(final_results),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
        except BadRequest:
            raise
        except Exception as e:
            self.logger.error(f"❌ Text analysis and search failed: {e}")
            raise InternalServerError(f"Text analysis and search failed: {str(e)}")
    
    def _analyze_text_with_llm(self, text: str) -> Optional[Dict]:
        """Analyze text using Azure OpenAI to extract user scenarios"""
        try:
            prompt = TEXT_ANALYSIS_PROMPT_TEMPLATE.format(text=text)
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": TEXT_ANALYSIS_SYSTEM_MESSAGE
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get("temperature", 1),
                max_tokens=self.config.get("max_tokens", 800)
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting that might be present
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            analysis = json.loads(content)
            
            # Validate and clean the response
            analysis = self._validate_and_clean_analysis(analysis, 'usage_scenarios')
            
            return analysis
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
            self.logger.error(f"Raw response: {content}")
            return None
        except Exception as e:
            self.logger.error(f"❌ LLM analysis failed: {e}")
            return None

    def _validate_and_clean_analysis(self, analysis: Dict, key: str) -> Dict:
        """Validate and clean the analysis results from LLM"""
        # Ensure required fields exist with defaults
        cleaned = {
            key: analysis.get(key, []),
        }

        # Ensure context_suggestions is actually a list
        if not isinstance(cleaned[key], list):
            cleaned[key] = [str(cleaned[key])] if cleaned[key] else []
        
        return cleaned
    
    def _create_search_query_from_analysis(self, analysis: Dict) -> str:
        """Create a search query from the analysis results"""
        query_parts = []
        
        # Add usage scenarios
        usage_scenarios = analysis.get('usage_scenarios', [])
        query_parts.append(EmojiVectorStorage.convert_to_string(usage_scenarios)) 

        # Create search query
        search_query = ' '.join(part for part in query_parts if part)

        return search_query
    
    # ...existing methods...


def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_emoji_name_from_filename(filename: str) -> str:
    """Extract readable name from filename"""
    # Remove common image extensions and replace underscores with spaces
    name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.gif', '').replace('.bmp', '').replace('.webp', '').replace('_', ' ')
    
    # Handle unicode hex codes
    if name.startswith(('1f', '2', '00')):
        parts = name.split(' ')
        if len(parts) > 1:
            return ' '.join(parts[1:]).title()
    
    return name.title()


def convert_image_to_png(input_path: str, output_path: str) -> bool:
    """Convert any supported image format to PNG format"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL not available, skipping image conversion")
            return False
            
        # Open the image with PIL
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (for formats like WEBP with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Keep transparency for PNG
                if img.mode == 'P':
                    img = img.convert('RGBA')
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Save as PNG with optimization
            img.save(output_path, 'PNG', optimize=True)
            
        logger.info(f"✅ Successfully converted image to PNG: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to convert image to PNG: {e}")
        return False


def analyze_uploaded_image_with_azure_openai(image_path: str, config: Dict) -> Optional[Dict]:
    """Analyze uploaded image using Azure OpenAI Vision API"""
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=config.get('azure_openai_api_key'),
            api_version=config.get('azure_openai_api_version', '2024-02-15-preview'),
            azure_endpoint=config.get('azure_openai_endpoint')
        )
        
        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = EMOJI_ANALYSIS_PROMPT
        
        response = client.chat.completions.create(
            model=config.get('deployment_name', 'gpt-35-turbo'),
            messages=[
                {
                    "role": "system", 
                    "content": EMOJI_ANALYSIS_SYSTEM_MESSAGE
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 800)
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up any markdown formatting
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
        
    except FileNotFoundError:
        logger.error(f"❌ Image file not found: {image_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {e}")
        logger.error(f"Raw response: {content}")
        return None
    except Exception as e:
        logger.error(f"❌ Azure OpenAI Vision API error: {e}")
        return None


def store_analysis_in_vector_db(emoji_key: str, analysis_data: Dict, storage_instance: EmojiSearchAPI) -> bool:
    """Store the analysis results in Azure AI Search vector database"""
    try:
        # Create a document similar to how emoji_vector_storage.py does it
        from emoji_vector_storage import EmojiDocument
        
        # Convert analysis data to searchable text
        searchable_text_parts = [
            analysis_data.get('primary_emotion', ''),
            ' '.join(analysis_data.get('secondary_emotions', [])),
            ' '.join(analysis_data.get('usage_scenarios', [])),
            analysis_data.get('tone', ''),
            ' '.join(analysis_data.get('context_suggestions', []))
        ]
        searchable_text = ' '.join(part for part in searchable_text_parts if part).strip()
        
        # Generate embedding for the searchable text
        embedding = storage_instance.storage.generate_embedding(searchable_text)
        if not embedding:
            logger.error(f"❌ Failed to generate embedding for {emoji_key}")
            return False
        
        # Create document
        document = EmojiDocument(
            id=emoji_key,
            emoji_name=analysis_data.get('emoji_name', ''),
            filename=analysis_data.get('filename', ''),
            primary_emotion=analysis_data.get('primary_emotion', ''),
            secondary_emotions=storage_instance.storage.convert_to_string(analysis_data.get('secondary_emotions', [])),
            usage_scenarios=storage_instance.storage.convert_to_string(analysis_data.get('usage_scenarios', [])),
            tone=analysis_data.get('tone', ''),
            context_suggestions=storage_instance.storage.convert_to_string(analysis_data.get('context_suggestions', [])),
            content_vector=embedding,
            searchable_text=searchable_text
        )
        
        # Upload document to vector database
        success = storage_instance.storage.upload_documents([document], batch_size=1)
        
        if success:
            logger.info(f"✅ Successfully stored analysis for {emoji_key} in vector database")
        else:
            logger.error(f"❌ Failed to store analysis for {emoji_key} in vector database")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error storing analysis in vector database: {e}")
        return False


def delete_emoji_by_name(filename: str, storage_instance: EmojiSearchAPI) -> Dict[str, Any]:
    """Delete emoji file and its database entry by filename"""
    try:
        # Normalize filename to ensure it ends with .png
        if not filename.endswith('.png'):
            # Try common extensions
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            filename = f"{base_name}.png"
        
        # Full path to the image file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        file_deleted = False
        
        if file_exists:
            try:
                os.remove(file_path)
                file_deleted = True
                logger.info(f"🗑️ Successfully deleted file: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to delete file {filename}: {e}")
                file_deleted = False
        else:
            logger.warning(f"⚠️ File not found: {filename}")
        
        # Delete from vector database
        database_deleted = storage_instance.storage.delete_emoji_by_filename(filename)
        
        # Determine overall success
        success = database_deleted or file_deleted
        
        result = {
            "success": success,
            "filename": filename,
            "file_existed": file_exists,
            "file_deleted": file_deleted,
            "database_deleted": database_deleted,
            "message": ""
        }
        
        # Generate appropriate message
        if file_deleted and database_deleted:
            result["message"] = "Emoji file and database entry deleted successfully"
        elif file_deleted and not database_deleted:
            result["message"] = "Emoji file deleted, but database entry not found or failed to delete"
        elif not file_deleted and database_deleted:
            result["message"] = "Database entry deleted, but file not found or failed to delete"
        elif not file_exists and not database_deleted:
            result["message"] = "Emoji not found in file system or database"
            result["success"] = False
        else:
            result["message"] = "Failed to delete emoji file and database entry"
            result["success"] = False
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error deleting emoji {filename}: {e}")
        return {
            "success": False,
            "filename": filename,
            "file_existed": False,
            "file_deleted": False,
            "database_deleted": False,
            "message": f"Error during deletion: {str(e)}"
        }


def delete_emoji_index_by_name(filename: str, storage_instance: EmojiSearchAPI) -> Dict[str, Any]:
    """Delete emoji index from vector database only (keeps file intact) by filename"""
    try:
        # Normalize filename to ensure it ends with .png
        if not filename.endswith('.png'):
            # Try common extensions
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            filename = f"{base_name}.png"
        
        # Full path to check if file still exists (for informational purposes)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_exists = os.path.exists(file_path)
        
        # Delete from vector database only
        database_deleted = storage_instance.storage.delete_emoji_by_filename(filename)
        
        result = {
            "success": database_deleted,
            "filename": filename,
            "file_exists": file_exists,
            "file_kept_intact": True,  # We never touch the file
            "database_deleted": database_deleted,
            "message": ""
        }
        
        # Generate appropriate message
        if database_deleted:
            if file_exists:
                result["message"] = f"Index deleted successfully. File '{filename}' kept intact."
            else:
                result["message"] = f"Index deleted successfully. Note: File '{filename}' was not found on disk."
        else:
            result["message"] = f"Index entry for '{filename}' not found in database or failed to delete"
            result["success"] = False
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error deleting emoji index {filename}: {e}")
        return {
            "success": False,
            "filename": filename,
            "file_exists": False,
            "file_kept_intact": True,
            "database_deleted": False,
            "message": f"Error during index deletion: {str(e)}"
        }


def initialize_storage():
    """Initialize the global storage instance"""
    global storage
    try:
        config_file = "azure_config.json"
        if not os.path.exists(config_file):
            logger.error(f"❌ Configuration file '{config_file}' not found!")
            return False
        
        storage = EmojiSearchAPI(config_file)
        
        # Test if database has data
        stats = storage.get_stats()
        if not stats or stats.get('total_count', 0) == 0:
            logger.error("❌ No emoji data found in the vector database!")
            return False
        
        logger.info(f"✅ Connected to vector database with {stats['total_count']} emojis")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing storage: {e}")
        return False


def require_storage():
    """Decorator to ensure storage is initialized"""
    def decorator(f):
        def wrapper(*args, **kwargs):
            if storage is None:
                return jsonify({
                    "error": "Service unavailable",
                    "message": "Database not initialized"
                }), 503
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


# API Routes
@app.route('/')
def index():
    """API documentation root"""
    return render_template('api_docs.html')


@app.route('/search')
def search_interface():
    """Interactive web search interface"""
    return render_template('search_interface.html')


@app.route('/upload')
def upload_interface():
    """Interactive web upload interface"""
    return render_template('upload_interface.html')


@app.route('/static/downloaded_emojis/<filename>')
def static_files(filename):
    """Serve static files (emoji images)"""
    try:
        # Serve files from the downloaded_emojis directory
        return send_from_directory('downloaded_emojis', filename)
    except Exception as e:
        logger.error(f"❌ Error serving static file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database_initialized": storage is not None
    })

@app.route('/api/stats')
@require_storage()
def stats():
    """Get database statistics"""
    try:
        result = storage.get_stats()
        return jsonify(result)
        
    except InternalServerError as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error in stats: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/search-by-name', methods=['GET', 'POST'])
@require_storage()
def search_emoji_by_name():
    """Search for emoji by name"""
    try:
        if request.method == 'GET':
            emoji_name = request.args.get('name', '').strip()
        else:  # POST
            data = request.get_json()
            if not data:
                raise BadRequest("JSON payload required for POST requests")
            emoji_name = data.get('name', '').strip()
        
        if not emoji_name:
            raise BadRequest("Emoji name is required")
        
        result = storage.search_by_name(emoji_name)
        return jsonify(result)
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except InternalServerError as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error in name search: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/search-by-filename', methods=['GET', 'POST'])
@require_storage()
def search_emoji_by_filename():
    """Search for emoji index by filename containing the query"""
    try:
        if request.method == 'GET':
            filename = request.args.get('filename', '').strip()
            top_k = int(request.args.get('top_k', 10))
        else:  # POST
            data = request.get_json()
            if not data:
                raise BadRequest("JSON payload required for POST requests")
            filename = data.get('filename', '').strip()
            top_k = int(data.get('top_k', 10))
        
        if not filename:
            raise BadRequest("Filename is required")
        
        if top_k < 1 or top_k > 100:
            raise BadRequest("top_k must be between 1 and 100")
        
        result = storage.search_by_filename(filename, top_k)
        return jsonify(result)
        
    except ValueError:
        return jsonify({"error": "Bad Request", "message": "top_k must be a valid integer"}), 400
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except InternalServerError as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error in filename search: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/search-by-filename/<filename>', methods=['GET'])
@require_storage()
def search_emoji_by_filename_path(filename):
    """Search for emoji index by filename containing the query (path parameter version)"""
    try:
        if not filename.strip():
            raise BadRequest("Filename is required")
        
        top_k = int(request.args.get('top_k', 10))
        
        if top_k < 1 or top_k > 100:
            raise BadRequest("top_k must be between 1 and 100")
        
        result = storage.search_by_filename(filename, top_k)
        return jsonify(result)
        
    except ValueError:
        return jsonify({"error": "Bad Request", "message": "top_k must be a valid integer"}), 400
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except InternalServerError as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error in filename search: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/analyze-text', methods=['POST'])
@require_storage()
def analyze_text():
    """Analyze text using LLM to extract emotions and context, then search for matching emojis"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("JSON payload required")
        
        text = data.get('text', '').strip()
        top_k = data.get('top_k', 5)
        
        result = storage.analyze_text_and_search(text, top_k)
        return jsonify(result)
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except InternalServerError as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error in text analysis: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/upload-analyze', methods=['POST'])
@require_storage()
def upload_analyze():
    """Upload an image, analyze it with Azure OpenAI Vision, and store results in vector database"""
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            raise BadRequest("No file uploaded. Please provide a file in the 'file' field.")
        
        file = request.files['file']
        if file.filename == '':
            raise BadRequest("No file selected")
        
        # Validate file type
        if not allowed_file(file.filename):
            raise BadRequest(f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
        
        # Generate filename that preserves original name but ensures uniqueness
        original_filename = secure_filename(file.filename)
        original_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'png'
        
        # Extract base name without extension
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
        
        # Create PNG filename preserving original base name
        png_filename = f"{base_name}.png"
        png_file_path = os.path.join(UPLOAD_FOLDER, png_filename)
        
        # Handle filename conflicts by adding counter
        counter = 1
        while os.path.exists(png_file_path):
            png_filename = f"{base_name}_{counter}.png"
            png_file_path = os.path.join(UPLOAD_FOLDER, png_filename)
            counter += 1
        
        # If the original file is already PNG, save directly
        if original_extension == 'png':
            file.save(png_file_path)
            logger.info(f"📤 PNG file uploaded successfully: {png_filename}")
        else:
            # For non-PNG files, save temporarily and convert
            temp_filename = f"temp_{base_name}_{int(time.time())}.{original_extension}"
            temp_file_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            
            try:
                # Save the original file temporarily
                file.save(temp_file_path)
                logger.info(f"📤 File uploaded temporarily: {temp_filename}")
                
                # Convert to PNG
                if convert_image_to_png(temp_file_path, png_file_path):
                    logger.info(f"🔄 Successfully converted {original_extension.upper()} to PNG: {png_filename}")
                else:
                    # If conversion fails, keep original file but rename to indicate failure
                    os.rename(temp_file_path, png_file_path)
                    logger.warning(f"⚠️ Conversion failed, keeping original format: {png_filename}")
                    
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except:
                    pass
                    
            except Exception as e:
                # Clean up temporary file on error
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except:
                    pass
                raise InternalServerError(f"Failed to process uploaded file: {str(e)}")
        
        # Use the PNG filename and path for further processing
        filename = png_filename
        file_path = png_file_path
        
        # Load configuration for Azure OpenAI
        config_file = "azure_config.json"
        if not os.path.exists(config_file):
            raise InternalServerError("Azure configuration not found")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Analyze the uploaded image using Azure OpenAI Vision
        logger.info(f"🔍 Analyzing image: {filename}")
        analysis_result = analyze_uploaded_image_with_azure_openai(file_path, config)
        
        if not analysis_result:
            # Clean up the uploaded file if analysis failed
            try:
                os.remove(file_path)
            except:
                pass
            raise InternalServerError("Failed to analyze the uploaded image")
        
        # Extract emoji name from filename
        emoji_name = extract_emoji_name_from_filename(filename)
        
        # Add metadata to analysis result
        analysis_result['filename'] = filename
        analysis_result['emoji_name'] = emoji_name
        analysis_result['upload_timestamp'] = datetime.utcnow().isoformat() + "Z"
        analysis_result['file_path'] = file_path
        
        # Create unique key for the vector database
        emoji_key = filename.rsplit('.', 1)[0]  # Remove file extension
        
        # Store the analysis in the vector database
        logger.info(f"💾 Storing analysis results in vector database: {emoji_key}")
        storage_success = store_analysis_in_vector_db(emoji_key, analysis_result, storage)
        
        if not storage_success:
            logger.warning(f"⚠️ Analysis completed but failed to store in vector database")
        
        # Prepare response
        response_data = {
            "success": True,
            "filename": filename,
            "emoji_name": emoji_name,
            "analysis": analysis_result,
            "stored_in_database": storage_success,
            "message": "Image uploaded and analyzed successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if storage_success:
            response_data["message"] += " and stored in vector database"
            
            # Refresh database statistics after successful storage
            try:
                updated_stats = storage.get_stats()
                response_data["database_stats"] = updated_stats
                logger.info(f"📊 Database statistics refreshed: {updated_stats.get('total_count', 0)} total emojis")
            except Exception as e:
                logger.warning(f"⚠️ Failed to refresh database statistics: {e}")
                # Don't fail the request if stats refresh fails
                response_data["stats_refresh_error"] = str(e)
        
        logger.info(f"✅ Upload and analysis completed: {filename}")
        return jsonify(response_data)
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except InternalServerError as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error in upload analysis: {e}")
        # Clean up uploaded file if it exists
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/delete-emoji', methods=['DELETE'])
@require_storage()
def delete_emoji():
    """Delete an emoji by filename from both filesystem and database"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("JSON payload required")
        
        filename = data.get('filename', '').strip()
        if not filename:
            raise BadRequest("Filename is required")
        
        # Perform deletion
        result = delete_emoji_by_name(filename, storage)
        
        if result["success"]:
            # Refresh database statistics after successful deletion
            try:
                updated_stats = storage.get_stats()
                result["database_stats"] = updated_stats
                logger.info(f"📊 Database statistics refreshed after deletion: {updated_stats.get('total_count', 0)} total emojis")
            except Exception as e:
                logger.warning(f"⚠️ Failed to refresh database statistics after deletion: {e}")
                # Don't fail the request if stats refresh fails
                result["stats_refresh_error"] = str(e)
            
            logger.info(f"✅ Emoji deletion completed: {filename}")
            return jsonify(result)
        else:
            logger.warning(f"⚠️ Emoji deletion failed: {filename}")
            return jsonify(result), 404 if "not found" in result["message"].lower() else 400
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Unexpected error in emoji deletion: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/delete-emoji/<filename>', methods=['DELETE'])
@require_storage()
def delete_emoji_by_path(filename):
    """Delete an emoji by filename from both filesystem and database (path parameter version)"""
    try:
        if not filename.strip():
            raise BadRequest("Filename is required")
        
        # Perform deletion
        result = delete_emoji_by_name(filename, storage)
        
        if result["success"]:
            # Refresh database statistics after successful deletion
            try:
                updated_stats = storage.get_stats()
                result["database_stats"] = updated_stats
                logger.info(f"📊 Database statistics refreshed after deletion: {updated_stats.get('total_count', 0)} total emojis")
            except Exception as e:
                logger.warning(f"⚠️ Failed to refresh database statistics after deletion: {e}")
                # Don't fail the request if stats refresh fails
                result["stats_refresh_error"] = str(e)
            
            logger.info(f"✅ Emoji deletion completed: {filename}")
            return jsonify(result)
        else:
            logger.warning(f"⚠️ Emoji deletion failed: {filename}")
            return jsonify(result), 404 if "not found" in result["message"].lower() else 400
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Unexpected error in emoji deletion: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/delete-index', methods=['DELETE'])
@require_storage()
def delete_emoji_index():
    """Delete an emoji index from vector database only (keeps file intact)"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("JSON payload required")
        
        filename = data.get('filename', '').strip()
        if not filename:
            raise BadRequest("Filename is required")
        
        # Perform database deletion only
        result = delete_emoji_index_by_name(filename, storage)
        
        if result["success"]:
            # Refresh database statistics after successful index deletion
            try:
                updated_stats = storage.get_stats()
                result["database_stats"] = updated_stats
                logger.info(f"📊 Database statistics refreshed after index deletion: {updated_stats.get('total_count', 0)} total emojis")
            except Exception as e:
                logger.warning(f"⚠️ Failed to refresh database statistics after index deletion: {e}")
                # Don't fail the request if stats refresh fails
                result["stats_refresh_error"] = str(e)
            
            logger.info(f"✅ Emoji index deletion completed: {filename}")
            return jsonify(result)
        else:
            logger.warning(f"⚠️ Emoji index deletion failed: {filename}")
            return jsonify(result), 404 if "not found" in result["message"].lower() else 400
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Unexpected error in emoji index deletion: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.route('/api/delete-index/<filename>', methods=['DELETE'])
@require_storage()
def delete_emoji_index_by_path(filename):
    """Delete an emoji index from vector database only (keeps file intact) - path parameter version"""
    try:
        if not filename.strip():
            raise BadRequest("Filename is required")
        
        # Perform database deletion only
        result = delete_emoji_index_by_name(filename, storage)
        
        if result["success"]:
            # Refresh database statistics after successful index deletion
            try:
                updated_stats = storage.get_stats()
                result["database_stats"] = updated_stats
                logger.info(f"📊 Database statistics refreshed after index deletion: {updated_stats.get('total_count', 0)} total emojis")
            except Exception as e:
                logger.warning(f"⚠️ Failed to refresh database statistics after index deletion: {e}")
                # Don't fail the request if stats refresh fails
                result["stats_refresh_error"] = str(e)
            
            logger.info(f"✅ Emoji index deletion completed: {filename}")
            return jsonify(result)
        else:
            logger.warning(f"⚠️ Emoji index deletion failed: {filename}")
            return jsonify(result), 404 if "not found" in result["message"].lower() else 400
        
    except BadRequest as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Unexpected error in emoji index deletion: {e}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint was not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        "error": "Method Not Allowed",
        "message": "The requested method is not allowed for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500


def create_app():
    """Application factory"""
    if not initialize_storage():
        logger.error("❌ Failed to initialize storage. Some endpoints may not work.")
    
    return app


if __name__ == "__main__":
    print("🎭 Emoji Search API Server")
    print("=" * 50)
    
    # Check configuration
    config_file = "azure_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found!")
        print("Please ensure azure_config.json exists with proper Azure credentials.")
        exit(1)
    
    # Initialize the application
    app = create_app()
    
    # Start the server
    print("🚀 Starting Flask server...")
    print("🎭 Interactive Search: http://localhost:5000/search")
    print("📤 Upload & Analyze: http://localhost:5000/upload")
    print("📖 API Documentation: http://localhost:5000/")
    print("🔍 Health Check: http://localhost:5000/api/health")
    print("=" * 50)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
