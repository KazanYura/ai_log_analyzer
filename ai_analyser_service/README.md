# RAG-based AI Analyzer Service

## Overview
This service has been enhanced with a **RAG (Retrieval-Augmented Generation)** architecture to better handle large volumes of log data. The RAG approach combines:

1. **Knowledge Base**: Stores and indexes log patterns with their solutions
2. **Retrieval System**: Finds relevant patterns based on current log events
3. **Generation System**: Creates contextualized analysis using retrieved information

## Architecture Components

### 1. LogPattern Class
- Represents reusable log patterns with metadata
- Includes severity, category, solution, and frequency information
- Automatically generates pattern hashes for efficient storage

### 2. RAGKnowledgeBase Class
- Stores and manages log patterns
- Uses embeddings for semantic similarity matching
- Supports both sentence-transformers and fallback similarity methods
- Automatically updates when new patterns are discovered

### 3. RAGAIAnalyzerService Class
- Main service class with RAG capabilities
- Extracts patterns from incoming log events
- Retrieves relevant context from knowledge base
- Generates analysis using both retrieved patterns and current events

## Key Features

### 🚀 **Scalability**
- Handles massive log datasets efficiently
- Uses semantic search instead of processing all logs
- Incremental learning from new patterns

### 🧠 **Intelligence**
- Learns from historical patterns
- Provides contextual solutions based on similar past issues
- Continuously improves recommendations

### 🔄 **Fallback System**
- Graceful degradation when AI models fail
- Rule-based analysis for reliable results
- Optional dependency on sentence-transformers

### 🏗️ **Backward Compatibility**
- Original AIAnalyzerService still available
- Automatically delegates to RAG service for better performance

## Usage

### Basic Usage
```python
from src.services.ai_service import RAGAIAnalyzerService

# Initialize RAG service
rag_service = RAGAIAnalyzerService()

# Analyze log events
analysis = rag_service.generate_advice(log_events)
```

### Large Dataset Handling
The service automatically:
- Extracts patterns from current events
- Retrieves similar historical patterns
- Generates contextual analysis
- Falls back to rule-based analysis if needed

## Benefits for Large Log Volumes

### 1. **Efficient Processing**
- Only processes relevant patterns, not all logs
- Semantic search reduces computational overhead
- Incremental pattern extraction

### 2. **Better Accuracy**
- Leverages historical knowledge
- Provides specific solutions for known issues
- Reduces false positives

### 3. **Continuous Learning**
- Automatically discovers new patterns
- Updates frequency and source information
- Improves recommendations over time

### 4. **Robust Fallback**
- Rule-based analysis when AI fails
- Graceful handling of missing dependencies
- Consistent output format

## Dependencies

### Required
- `transformers`: For language model and tokenization
- `numpy`: For numerical operations
- `scikit-learn`: For cosine similarity (when sentence-transformers unavailable)

### Optional
- `sentence-transformers`: For better semantic embeddings
- `sklearn`: Enhanced similarity calculations

## Pattern Categories

The system automatically categorizes patterns into:
- **Network**: Connection timeouts, socket errors
- **Resource**: Memory issues, disk space
- **Security**: Authentication, permissions
- **Database**: Connection failures, query errors
- **CI/CD**: Build failures, deployment issues
- **Testing**: Test failures, assertion errors
- **Service**: API errors, service unavailability
- **General**: Uncategorized issues

## Example Output

```
=== RAG-BASED LOG ANALYSIS ===

1. ISSUES FOUND:
   - 5 CRITICAL issues requiring immediate attention
   - 23 ERROR events affecting system functionality
   - Found matches with known patterns: "Database connection failed"

2. PATTERNS OBSERVED:
   - Match with historical pattern: Connection timeout (occurred 15 times before)
   - New pattern discovered: Service restart loop
   - High frequency in CI/CD category

3. SEVERITY ASSESSMENT:
   - Overall Health: HIGH
   - Database connectivity issues are critical
   - Service instability detected

4. RECOMMENDATIONS:
   - Apply known solution: Check database service status, verify connection strings
   - Monitor service restart frequency
   - Implement circuit breaker pattern for database connections
```

## Configuration

The service can be configured with different models:

```python
# Use different language model
rag_service = RAGAIAnalyzerService(model_name="microsoft/DialoGPT-small")

# Access knowledge base directly
patterns = rag_service.knowledge_base.retrieve_similar_patterns("connection timeout")
```

This RAG architecture makes the service much more capable of handling large log volumes while providing intelligent, context-aware analysis.
