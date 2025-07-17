# AI Log Analyzer

A comprehensive microservices-based log analysis system that leverages AI and RAG (Retrieval-Augmented Generation) architecture to provide intelligent log analysis and troubleshooting recommendations.

## Architecture Overview

This project consists of three main microservices:

### 1. Parser Service (Port 8001)
- Parses various log formats (currently supports CircleCI)
- Extracts structured log events from raw text
- Provides health check endpoints
- **Technologies**: FastAPI, Python 3.10

### 2. AI Analyzer Service (Port 8000)
- RAG-based intelligent log analysis
- Pattern recognition and knowledge base
- AI-powered troubleshooting recommendations
- **Technologies**: FastAPI, Transformers, Sentence-Transformers, scikit-learn

### 3. Orchestrator Service (Port 8002)
- Coordinates between parser and AI analyzer services
- Manages service communication
- Provides unified API endpoints
- **Technologies**: FastAPI, Python 3.10

## Key Features

### 🤖 **AI-Powered Analysis**
- Uses Microsoft DialoGPT for intelligent response generation
- Implements RAG architecture for context-aware analysis
- Automatic pattern recognition and categorization

### 📊 **Advanced Pattern Recognition**
- Semantic similarity matching using sentence-transformers
- Fallback to token-based similarity for robustness
- Continuous learning from log patterns

### 🔄 **Microservices Architecture**
- Scalable and maintainable service-oriented design
- Independent deployment and scaling
- Docker containerization for easy deployment

### 🛡️ **Robust Fallback System**
- Multiple fallback mechanisms for reliability
- Graceful degradation when AI models are unavailable
- Structured analysis when generation fails

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local development)
- Poetry (for dependency management)

### Running with Docker

1. **Build and run all services:**
```bash
docker-compose up --build
```

2. **Access the services:**
- Parser Service: http://localhost:8001
- AI Analyzer Service: http://localhost:8000
- Orchestrator Service: http://localhost:8002

### API Endpoints

#### Parser Service
- `POST /parse_log/` - Parse log files
- `GET /health` - Health check

#### AI Analyzer Service
- `POST /generate-advice` - Generate AI-powered analysis
- `GET /health` - Health check

#### Orchestrator Service
- Coordinates service interactions
- `GET /health` - Health check

## Development

### Local Development Setup

1. **Install dependencies for each service:**
```bash
cd ai_analyser_service && poetry install
cd ../parser_service && poetry install
cd ../orchestrator_service && poetry install
```

2. **Run services locally:**
```bash
# Terminal 1 - Parser Service
cd parser_service
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8001

# Terminal 2 - AI Analyzer Service
cd ai_analyser_service
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Terminal 3 - Orchestrator Service
cd orchestrator_service
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8002
```

## Configuration

### AI Models
- **Primary**: Microsoft DialoGPT-medium for text generation
- **Embeddings**: all-MiniLM-L6-v2 for semantic similarity
- **Fallback**: BERT tokenizer for basic similarity

### Environment Variables
- `PYTHONPATH`: Set to `/app/src` in containers
- Model configurations are handled in `config/config.py`

## Supported Log Formats

Currently supported:
- **CircleCI**: Complete parsing support with error extraction
- **Extensible**: Easy to add new log format parsers

## Technology Stack

- **Backend**: FastAPI, Python 3.10
- **AI/ML**: Transformers, Sentence-Transformers, scikit-learn
- **Deployment**: Docker, Docker Compose
- **Package Management**: Poetry
- **Architecture**: Microservices, RAG (Retrieval-Augmented Generation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is available under the MIT License.
