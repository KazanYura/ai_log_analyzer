from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import hashlib
import re
from datetime import datetime
from src.models.models import LogEvent, LogLevel
from src.config.config import DEFAULT_MODEL_NAME
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False




class LogPattern:
    """Represents a log pattern with metadata for RAG retrieval"""
    def __init__(self, pattern: str, severity: LogLevel, category: str, 
                 solution: str, frequency: int = 1, sources: List[str] = None):
        self.pattern = pattern
        self.severity = severity
        self.category = category
        self.solution = solution
        self.frequency = frequency
        self.sources = sources or []
        self.embedding = None
        self.pattern_hash = hashlib.md5(pattern.encode()).hexdigest()
        self.last_seen = datetime.now()

class RAGKnowledgeBase:
    """Knowledge base for storing and retrieving log patterns"""
    def __init__(self):
        self.patterns: Dict[str, LogPattern] = {}
        self.embeddings = []
        self.pattern_keys = []
        self.embedding_model = None
        self._load_embedding_model()
        
    def _load_embedding_model(self):
        """Load a lightweight embedding model for pattern matching"""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_sentence_transformers = True
            except Exception:
                self.embedding_model = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.use_sentence_transformers = False
        else:
            self.embedding_model = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.use_sentence_transformers = False
            
    def add_pattern(self, pattern: LogPattern):
        """Add or update a pattern in the knowledge base"""
        if pattern.pattern_hash in self.patterns:
            existing = self.patterns[pattern.pattern_hash]
            existing.frequency += pattern.frequency
            existing.sources.extend(pattern.sources)
            existing.sources = list(set(existing.sources))
            existing.last_seen = datetime.now()
        else:
            self.patterns[pattern.pattern_hash] = pattern
            self._update_embeddings()
            
    def _update_embeddings(self):
        """Update embeddings for all patterns"""
        self.pattern_keys = list(self.patterns.keys())
        pattern_texts = [self.patterns[key].pattern for key in self.pattern_keys]
        
        if hasattr(self.embedding_model, 'encode'):
            self.embeddings = self.embedding_model.encode(pattern_texts)
        else:
            self.embeddings = []
            for text in pattern_texts:
                tokens = self.embedding_model.tokenize(text)
                embedding = [len(tokens), len(set(tokens))]
                self.embeddings.append(embedding)
            self.embeddings = np.array(self.embeddings)
            
    def retrieve_similar_patterns(self, query: str, top_k: int = 3) -> List[LogPattern]:
        """Retrieve similar patterns from the knowledge base"""
        if not self.embeddings or len(self.embeddings) == 0:
            return []
            
        try:
            if hasattr(self.embedding_model, 'encode'):
                query_embedding = self.embedding_model.encode([query])
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            else:
                query_tokens = self.embedding_model.tokenize(query)
                query_embedding = np.array([len(query_tokens), len(set(query_tokens))])
                similarities = []
                for emb in self.embeddings:
                    dot_product = np.dot(query_embedding, emb)
                    norm_a = np.linalg.norm(query_embedding)
                    norm_b = np.linalg.norm(emb)
                    similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                    similarities.append(similarity)
                similarities = np.array(similarities)
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.patterns[self.pattern_keys[i]] for i in top_indices if similarities[i] > 0.3]
            
        except Exception:
            return []


class RAGAIAnalyzerService:
    """RAG-based AI Analyzer Service for log analysis"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.knowledge_base = RAGKnowledgeBase()
        self._load_model()
        self._initialize_knowledge_base()
    
    def _load_model(self):
        """Load the language model for generation"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with common log patterns"""
        common_patterns = [
            LogPattern(
                pattern="Connection timeout",
                severity=LogLevel.ERROR,
                category="Network",
                solution="Check network connectivity, increase timeout values, verify service availability"
            ),
            LogPattern(
                pattern="Out of memory",
                severity=LogLevel.CRITICAL,
                category="Resource",
                solution="Increase memory allocation, check for memory leaks, optimize memory usage"
            ),
            LogPattern(
                pattern="Authentication failed",
                severity=LogLevel.ERROR,
                category="Security",
                solution="Verify credentials, check authentication service, review security policies"
            ),
            LogPattern(
                pattern="Database connection failed",
                severity=LogLevel.CRITICAL,
                category="Database",
                solution="Check database service status, verify connection strings, review database logs"
            ),
            LogPattern(
                pattern="Build failed",
                severity=LogLevel.ERROR,
                category="CI/CD",
                solution="Review build logs, check dependencies, verify build configuration"
            ),
            LogPattern(
                pattern="Test failed",
                severity=LogLevel.WARNING,
                category="Testing",
                solution="Review test results, check test environment, update test cases if needed"
            ),
            LogPattern(
                pattern="Permission denied",
                severity=LogLevel.ERROR,
                category="Security",
                solution="Check file permissions, review user access rights, verify service account permissions"
            ),
            LogPattern(
                pattern="Service unavailable",
                severity=LogLevel.CRITICAL,
                category="Service",
                solution="Check service status, verify health endpoints, review service dependencies"
            )
        ]
        
        for pattern in common_patterns:
            self.knowledge_base.add_pattern(pattern)
    
    def _extract_patterns_from_events(self, events: List[LogEvent]) -> List[LogPattern]:
        """Extract patterns from log events and add to knowledge base"""
        patterns = []
        
        message_groups = {}
        for event in events:
            pattern_key = self._generalize_message(event.message)
            
            if pattern_key not in message_groups:
                message_groups[pattern_key] = []
            message_groups[pattern_key].append(event)
        
        for pattern_key, group_events in message_groups.items():
            if len(group_events) > 1:
                pattern = LogPattern(
                    pattern=pattern_key,
                    severity=group_events[0].level,
                    category=self._categorize_message(pattern_key),
                    solution="Review logs for specific context and apply appropriate fixes",
                    frequency=len(group_events),
                    sources={e.source for e in group_events}
                )
                patterns.append(pattern)
                self.knowledge_base.add_pattern(pattern)
        
        return patterns
    
    def _generalize_message(self, message: str) -> str:
        """Generalize log message by removing specific details"""
        import re
        
        generalized = re.sub(r'\b\d+\b', '<number>', message)
        generalized = re.sub(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b', '<uuid>', generalized)
        generalized = re.sub(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b', '<timestamp>', generalized)
        generalized = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '<ip>', generalized)
        generalized = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '<email>', generalized)
        generalized = re.sub(r'\b/[a-zA-Z0-9/_.-]+\b', '<path>', generalized)
        
        return generalized.strip()
    
    def _categorize_message(self, message: str) -> str:
        """Categorize log message based on keywords"""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['connection', 'timeout', 'network', 'socket']):
            return "Network"
        elif any(keyword in message_lower for keyword in ['memory', 'heap', 'oom', 'resource']):
            return "Resource"
        elif any(keyword in message_lower for keyword in ['auth', 'login', 'credential', 'permission']):
            return "Security"
        elif any(keyword in message_lower for keyword in ['database', 'sql', 'query', 'db']):
            return "Database"
        elif any(keyword in message_lower for keyword in ['build', 'compile', 'deploy', 'ci/cd']):
            return "CI/CD"
        elif any(keyword in message_lower for keyword in ['test', 'assertion', 'mock']):
            return "Testing"
        elif any(keyword in message_lower for keyword in ['service', 'api', 'endpoint']):
            return "Service"
        else:
            return "General"
    
    def _retrieve_context(self, events: List[LogEvent], top_k: int = 5) -> List[LogPattern]:
        """Retrieve relevant patterns from knowledge base based on current events"""
        all_relevant_patterns = []
        
        unique_messages = {self._generalize_message(e.message) for e in events}
        
        for message in unique_messages:
            similar_patterns = self.knowledge_base.retrieve_similar_patterns(message, top_k=3)
            all_relevant_patterns.extend(similar_patterns)
        
        unique_patterns = {}
        for pattern in all_relevant_patterns:
            if pattern.pattern_hash not in unique_patterns:
                unique_patterns[pattern.pattern_hash] = pattern
        
        return sorted(unique_patterns.values(), key=lambda x: x.frequency, reverse=True)[:top_k]
    
    def _build_rag_prompt(self, events: List[LogEvent], retrieved_patterns: List[LogPattern]) -> str:
        """Build RAG prompt with retrieved context"""
        
        event_summary = self._summarize_events_for_rag(events)
        
        context_info = ""
        if retrieved_patterns:
            context_info = "RELEVANT PATTERNS FROM KNOWLEDGE BASE:\n"
            for i, pattern in enumerate(retrieved_patterns, 1):
                context_info += f"{i}. Pattern: {pattern.pattern}\n"
                context_info += f"   Category: {pattern.category}\n"
                context_info += f"   Severity: {pattern.severity}\n"
                context_info += f"   Solution: {pattern.solution}\n"
                context_info += f"   Frequency: {pattern.frequency} occurrences\n\n"
        
        prompt = f"""You are an expert log analyzer. Use the context below to analyze current log events.

{context_info}

CURRENT LOG EVENTS ({len(events)} total):
{event_summary}

Based on the patterns above and current events, provide:

1. ISSUES FOUND:
   - Specific problems identified
   - Severity and impact assessment

2. PATTERNS OBSERVED:
   - Match with known patterns
   - New patterns discovered
   - Frequency analysis

3. SEVERITY ASSESSMENT:
   - Overall system health (CRITICAL/HIGH/MEDIUM/LOW)
   - Risk factors

4. RECOMMENDATIONS:
   - Immediate actions needed
   - Preventive measures
   - Reference to known solutions

Analysis:
"""
        return prompt
    
    def _summarize_events_for_rag(self, events: List[LogEvent]) -> str:
        """Summarize events specifically for RAG processing"""
        if len(events) <= 20:
            return "\n".join(f"[{e.level}] {e.message} (from: {e.source})" for e in events)
        
        critical_events = [e for e in events if e.level == LogLevel.CRITICAL]
        error_events = [e for e in events if e.level == LogLevel.ERROR]
        warning_events = [e for e in events if e.level == LogLevel.WARNING]
        
        summary = []
        summary.append(f"Event Distribution: {len(critical_events)} CRITICAL, {len(error_events)} ERROR, {len(warning_events)} WARNING")
        
        if critical_events:
            summary.append("\nCRITICAL Events:")
            for event in critical_events[:50]:
                summary.append(f"  - {event.message} (from: {event.source})")
        
        if error_events:
            summary.append("\nERROR Events (sample):")
            for event in error_events[:100]:
                summary.append(f"  - {event.message} (from: {event.source})")
        
        if warning_events:
            summary.append("\nWARNING Events (sample):")
            for event in warning_events[:25]:
                summary.append(f"  - {event.message} (from: {event.source})")
        
        return "\n".join(summary)
    
    def generate_advice(self, events: List[LogEvent]) -> str:
        """Generate advice using RAG architecture"""
        
        self._extract_patterns_from_events(events)
        
        retrieved_patterns = self._retrieve_context(events)
        
        prompt = self._build_rag_prompt(events, retrieved_patterns)
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1000,
                padding=True
            )
            
            if inputs is None or 'input_ids' not in inputs:
                return self._fallback_analysis(events, retrieved_patterns)
            
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 400,
                min_length=inputs['input_ids'].shape[1] + 50,
                num_beams=3,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else self.tokenizer.eos_token_id
            )
            
            if outputs is None or len(outputs) == 0:
                return self._fallback_analysis(events, retrieved_patterns)
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            if len(generated_tokens) == 0:
                return self._fallback_analysis(events, retrieved_patterns)
            
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
            
            return generated_text if generated_text else self._fallback_analysis(events, retrieved_patterns)
            
        except Exception:
            return self._fallback_analysis(events, retrieved_patterns)
    
    def _fallback_analysis(self, events: List[LogEvent], retrieved_patterns: List[LogPattern]) -> str:
        """Fallback analysis when AI generation fails"""
        
        analysis = ["=== RAG-BASED LOG ANALYSIS ===\n"]
        
        critical_events = [e for e in events if e.level == LogLevel.CRITICAL]
        error_events = [e for e in events if e.level == LogLevel.ERROR]
        warning_events = [e for e in events if e.level == LogLevel.WARNING]
        
        analysis.append("1. ISSUES FOUND:")
        if critical_events:
            analysis.append(f"   - {len(critical_events)} CRITICAL issues requiring immediate attention")
        if error_events:
            analysis.append(f"   - {len(error_events)} ERROR events affecting system functionality")
        if warning_events:
            analysis.append(f"   - {len(warning_events)} WARNING events indicating potential issues")
        
        analysis.append("\n2. PATTERNS OBSERVED:")
        if retrieved_patterns:
            analysis.append("   - Found matches with known patterns:")
            for pattern in retrieved_patterns[:3]:
                analysis.append(f"     * {pattern.pattern} (Category: {pattern.category})")
        
        analysis.append("\n3. SEVERITY ASSESSMENT:")
        if critical_events:
            analysis.append("   - Overall Health: CRITICAL")
        elif len(error_events) > len(events) * 0.1:
            analysis.append("   - Overall Health: HIGH")
        elif error_events:
            analysis.append("   - Overall Health: MEDIUM")
        else:
            analysis.append("   - Overall Health: LOW")
        
        analysis.append("\n4. RECOMMENDATIONS:")
        if retrieved_patterns:
            analysis.append("   - Apply known solutions:")
            for pattern in retrieved_patterns[:2]:
                analysis.append(f"     * {pattern.solution}")
        
        analysis.append("   - Monitor system performance and logs")
        analysis.append("   - Implement automated alerting for critical events")
        
        return "\n".join(analysis)
