# 🎭 Emoji Assistant

An intelligent emoji search and analysis system powered by Azure AI services. This application combines computer vision, natural language processing, and vector search to help you find the perfect emoji for any context.

## ✨ Features

### 🔍 Smart Emoji Search
- **Text-to-Emoji Analysis**: Analyze text sentiment and context to recommend relevant emojis
- **Filename Search**: Find emojis by partial filename matches
- **Name-based Search**: Search for specific emojis by name
- **Vector Similarity Search**: AI-powered semantic search using Azure AI Search

### 📤 Upload & Analysis
- **Image Upload & Analysis**: Upload emoji images for automatic emotion analysis
- **Vision AI Integration**: Uses Azure OpenAI Vision models (GPT-4o) for image understanding
- **Automatic Indexing**: Analyzed emojis are automatically stored in the vector database
- **Multiple Format Support**: Supports PNG, JPG, JPEG, GIF, BMP, and WEBP formats

### 🌐 Web Interface
- **Interactive Search Interface**: Web-based emoji search with real-time results
- **Upload Interface**: Drag-and-drop emoji upload with instant analysis
- **RESTful API**: Comprehensive REST API for all functionality
- **CORS Enabled**: Ready for frontend integration

### 🗄️ Vector Database
- **Azure AI Search Integration**: Scalable vector storage and retrieval
- **Embedding Generation**: Uses Azure OpenAI embeddings (text-embedding-ada-002)
- **Emotion Classification**: Primary emotions, secondary emotions, tone analysis
- **Usage Scenarios**: Context-aware usage suggestions

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   Flask API      │◄──►│  Azure AI       │
│  (HTML/JS/CSS)  │    │  (emoji_api.py)  │    │    Search       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Azure OpenAI    │
                        │  (Vision + LLM)  │
                        └──────────────────┘
```

### Key Components

- **`emoji_api.py`**: Flask web server providing RESTful API endpoints
- **`emoji_vector_storage.py`**: Azure AI Search integration and vector operations
- **`azure_emoji_analyzer.py`**: Azure OpenAI Vision API integration
- **`prompts.py`**: LLM prompts for emotion analysis
- **`templates/`**: Web interface HTML templates

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI account with GPT-4o (vision-enabled model)
- Azure AI Search service
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Emoji-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Azure services**
   
   Update `azure_config.json` with your Azure credentials:
   ```json
   {
     "azure_openai_api_key": "your-api-key-here",
     "azure_openai_endpoint": "https://your-resource.cognitiveservices.azure.com",
     "azure_openai_api_version": "2024-12-01-preview",
     "deployment_name": "gpt-4o",
     "search_service_name": "your-search-service",
     "search_admin_key": "your-search-admin-key",
     "embedding_deployment_name": "text-embedding-ada-002",
     "emoji_index_name": "emoji-emotions-from-vision"
   }
   ```

4. **Start the server**
   ```bash
   python emoji_api.py
   ```

5. **Access the application**
   - **Interactive Search**: http://localhost:5000/search
   - **Upload Interface**: http://localhost:5000/upload
   - **API Documentation**: http://localhost:5000/
   - **Health Check**: http://localhost:5000/api/health

## 📡 API Endpoints

### Text Analysis
```http
POST /api/analyze-text
Content-Type: application/json

{
  "text": "Happy birthday! Hope you have a wonderful day!",
  "top_k": 5
}
```

### Search by Filename
```http
GET /api/search-by-filename?filename=happy&top_k=10
```

### Search by Name
```http
GET /api/search-by-name?name=birthday
```

### Upload & Analyze
```http
POST /api/upload-analyze
Content-Type: multipart/form-data

file: [emoji image file]
```

### Delete Emoji
```http
DELETE /api/delete-emoji
Content-Type: application/json

{
  "filename": "custom_emoji.png"
}
```

### Statistics
```http
GET /api/stats
```

## 🎯 Usage Examples

### Text-to-Emoji Analysis
```python
import requests

response = requests.post("http://localhost:5000/api/analyze-text", json={
    "text": "Congratulations on your graduation!",
    "top_k": 3
})

print(response.json())
```

### Upload Custom Emoji
```python
import requests

with open("my_emoji.png", "rb") as f:
    response = requests.post(
        "http://localhost:5000/api/upload-analyze",
        files={"file": f}
    )

print(response.json())
```

## 🔧 Configuration

### Azure OpenAI Configuration
- **Model**: GPT-4o (vision-enabled) for image analysis
- **Embedding Model**: text-embedding-ada-002
- **API Version**: 2024-12-01-preview or later

### Azure AI Search Configuration
- **Index Name**: emoji-emotions-from-vision
- **Vector Dimensions**: 1536 (for text-embedding-ada-002)
- **Search Algorithm**: HNSW with cosine similarity

### Environment Variables (Optional)
```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

## 📁 Project Structure

```
Emoji-assistant/
├── emoji_api.py                 # Flask web server and API endpoints
├── emoji_vector_storage.py      # Azure AI Search integration
├── azure_emoji_analyzer.py      # Azure OpenAI Vision integration
├── emoji_downloader.py          # Utility for downloading emojis
├── emoji_search_interface.py    # Command-line search interface
├── prompts.py                   # LLM prompts for analysis
├── azure_config.json           # Azure service configuration
├── azure_emoji_emotions.json   # Emoji emotion data
├── requirements.txt             # Python dependencies
├── testquery.http              # API testing examples
├── templates/                   # Web interface templates
│   ├── api_docs.html
│   ├── search_interface.html
│   └── upload_interface.html
└── downloaded_emojis/          # Stored emoji images
    ├── happy.png
    ├── sad.png
    └── ...
```

## 🧪 Testing

Use the provided `testquery.http` file to test API endpoints:

```http
# Test text analysis
POST http://localhost:5000/api/analyze-text
Content-Type: application/json

{
  "text": "Merry Christmas and Happy New Year!",
  "top_k": 5
}

# Test filename search
GET http://localhost:5000/api/search-by-filename?filename=heart&top_k=10

# Test health check
GET http://localhost:5000/api/health
```

## 🔒 Security Considerations

- **API Keys**: Store Azure credentials securely, never commit to version control
- **File Uploads**: Validates file types and sizes to prevent malicious uploads
- **CORS**: Configured for cross-origin requests (adjust for production)
- **Rate Limiting**: Consider implementing rate limiting for production use

## 🚀 Deployment

### Local Development
```bash
python emoji_api.py
```

### Production Deployment
1. **Configure environment variables** for Azure credentials
2. **Set up reverse proxy** (nginx/Apache) for production
3. **Enable HTTPS** with SSL certificates
4. **Configure logging** for monitoring and debugging
5. **Set up auto-scaling** based on traffic patterns

### Docker Support (Optional)
Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "emoji_api.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Azure AI Services** for providing powerful AI capabilities
- **OpenAI** for the underlying language models
- **Flask** for the web framework
- **Community** for emoji assets and inspiration

## �� Support

For questions, issues, or contributions:

1. **GitHub Issues**: Report bugs or request features
2. **Documentation**: Check the inline code documentation
3. **API Testing**: Use the provided `testquery.http` examples

---

**Made with ❤️ and powered by Azure AI** 🚀
