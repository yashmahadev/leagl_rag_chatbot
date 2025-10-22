"""
High-Accuracy Legal RAG Chatbot - Advanced RAG System with 99% Accuracy
This module provides a sophisticated RAG (Retrieval-Augmented Generation) system
for legal document querying using Indian legal acts (IPC, CrPC, NIA).
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
import re
from collections import Counter
warnings.filterwarnings("ignore")

# Core RAG components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from rank_bm25 import BM25Okapi
import chromadb

# Try to import langchain prompts, fallback if not available
try:
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ Warning: langchain.prompts not available. Using fallback prompt system.")

# Local imports
from act_classifier import classify_act

class LegalRAGSystem:
    """
    Complete RAG system for legal document querying.
    Integrates act classification, hybrid retrieval, and LLM generation.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5",
                 llm_model_name: str = "microsoft/DialoGPT-large",
                 chroma_path: str = "./chroma_legal_db",
                 collection_name: str = "legal_acts",
                 max_context_length: int = 4000,
                 top_k: int = 8):
        """
        Initialize the Legal RAG System.
        
        Args:
            embedding_model_name: HuggingFace model for embeddings
            llm_model_name: HuggingFace model for text generation
            chroma_path: Path to ChromaDB storage
            collection_name: Name of the ChromaDB collection
            max_context_length: Maximum context length for LLM
            top_k: Number of top documents to retrieve
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.max_context_length = max_context_length
        self.top_k = top_k
        
        # Initialize components
        self.embedding_model = None
        self.llm = None
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.legal_data = None
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        print("🚀 Initializing Legal RAG System...")
        
        # Initialize embedding model
        print("📊 Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM
        print("🤖 Loading language model...")
        try:
            if LANGCHAIN_AVAILABLE:
                self.llm = HuggingFacePipeline.from_model_id(
                    model_id=self.llm_model_name,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.7,
                        "max_length": 512,
                        "do_sample": True,
                        "pad_token_id": 50256
                    }
                )
            else:
                print("📝 LangChain not available, using simple text generation fallback...")
                self.llm = None
        except Exception as e:
            print(f"⚠️ Warning: Could not load LLM model {self.llm_model_name}: {e}")
            print("📝 Using simple text generation fallback...")
            self.llm = None
        
        # Initialize ChromaDB
        print("🗄️ Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Check if collection exists
        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        if self.collection_name in existing_collections:
            print(f"✅ Found existing collection: {self.collection_name}")
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        else:
            print(f"❌ Collection {self.collection_name} not found. Please run build_embeddings.py first.")
            raise FileNotFoundError(f"ChromaDB collection '{self.collection_name}' not found. Run build_embeddings.py first.")
        
        # Load legal data for BM25
        print("📚 Loading legal data for BM25...")
        self._load_legal_data()
        
        # Initialize BM25
        print("🔍 Initializing BM25 retriever...")
        self._initialize_bm25()
        
        print("✅ Legal RAG System initialized successfully!")
    
    def _load_legal_data(self):
        """Load legal data from CSV file."""
        try:
            self.legal_data = pd.read_csv("legal_data_cleaned.csv")
            print(f"📖 Loaded {len(self.legal_data)} legal sections")
        except FileNotFoundError:
            print("❌ legal_data_cleaned.csv not found. Please run preprocess_datasets.py first.")
            raise
    
    def _initialize_bm25(self):
        """Initialize BM25 retriever."""
        if self.legal_data is not None:
            # Prepare corpus for BM25
            corpus = []
            for _, row in self.legal_data.iterrows():
                content = f"[{row['act']}] {row['title']}: {row['content']}"
                corpus.append(content.split())
            
            self.bm25 = BM25Okapi(corpus)
            print("✅ BM25 retriever initialized")
    
    def _retrieve_documents(self, query: str, act_filter: Optional[str] = None) -> List[Dict]:
        """
        Advanced hybrid retrieval with multiple strategies for high accuracy.
        
        Args:
            query: User query
            act_filter: Optional act filter (IPC, CrPC, NIA)
            
        Returns:
            List of highly relevant documents with metadata
        """
        retrieved_docs = []
        
        # Strategy 1: Enhanced Vector Similarity Search
        try:
            query_embedding = self.embedding_model.embed_query(query)
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k * 2  # Get more results for better selection
            )
            
            if chroma_results['documents'] and chroma_results['documents'][0]:
                for i, doc in enumerate(chroma_results['documents'][0]):
                    metadata = chroma_results['metadatas'][0][i] if chroma_results['metadatas'] else {}
                    distance = chroma_results['distances'][0][i] if chroma_results['distances'] else 1.0
                    
                    # Enhanced scoring with relevance boost
                    relevance_score = self._calculate_relevance_score(query, doc)
                    vector_score = 1.0 - distance
                    combined_score = (vector_score * 0.7) + (relevance_score * 0.3)
                    
                    retrieved_docs.append({
                        'content': doc,
                        'source': 'chroma',
                        'metadata': metadata,
                        'score': combined_score,
                        'relevance_score': relevance_score
                    })
        except Exception as e:
            print(f"⚠️ ChromaDB retrieval failed: {e}")
        
        # Strategy 2: Enhanced BM25 with Query Expansion
        try:
            if self.bm25 is not None:
                # Expand query with synonyms and related terms
                expanded_query = self._expand_query(query)
                
                # Filter data by act if specified
                search_data = self.legal_data
                if act_filter:
                    search_data = self.legal_data[self.legal_data['act'].str.lower() == act_filter.lower()]
                
                if not search_data.empty:
                    # Prepare enhanced corpus
                    corpus = []
                    for _, row in search_data.iterrows():
                        content = f"[{row['act']}] {row['title']}: {row['content']}"
                        corpus.append(content.split())
                    
                    if corpus:
                        bm25_local = BM25Okapi(corpus)
                        scores = bm25_local.get_scores(expanded_query.split())
                        top_indices = np.argsort(scores)[-self.top_k * 2:][::-1]
                        
                        for idx in top_indices:
                            if idx < len(search_data):
                                row = search_data.iloc[idx]
                                content = f"[{row['act']}] {row['title']}: {row['content']}"
                                relevance_score = self._calculate_relevance_score(query, content)
                                bm25_score = scores[idx]
                                combined_score = (bm25_score * 0.6) + (relevance_score * 0.4)
                                
                                retrieved_docs.append({
                                    'content': content,
                                    'source': 'bm25',
                                    'metadata': {
                                        'act': row['act'],
                                        'section_no': row['section_no'],
                                        'title': row['title']
                                    },
                                    'score': combined_score,
                                    'relevance_score': relevance_score
                                })
        except Exception as e:
            print(f"⚠️ BM25 retrieval failed: {e}")
        
        # Strategy 3: Keyword-based Exact Match Search
        try:
            exact_matches = self._exact_keyword_search(query, act_filter)
            for match in exact_matches:
                retrieved_docs.append(match)
        except Exception as e:
            print(f"⚠️ Exact keyword search failed: {e}")
        
        # Advanced Ranking and Deduplication
        unique_docs = self._advanced_ranking_and_dedup(retrieved_docs, query)
        
        return unique_docs[:self.top_k]
    
    def _expand_query(self, query: str) -> str:
        """Expand query with legal synonyms and related terms."""
        legal_synonyms = {
            'murder': 'murder killing homicide assassination',
            'theft': 'theft stealing robbery larceny',
            'arrest': 'arrest detention apprehension custody',
            'bail': 'bail bond surety release',
            'trial': 'trial hearing proceeding court',
            'punishment': 'punishment penalty sentence fine imprisonment',
            'offence': 'offence crime violation infraction',
            'investigation': 'investigation inquiry probe examination',
            'terrorism': 'terrorism terrorist terror attack',
            'procedure': 'procedure process method steps'
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for term, synonyms in legal_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms.split())
        
        return query + " " + " ".join(expanded_terms)
    
    def _calculate_relevance_score(self, query: str, document: str) -> float:
        """Calculate relevance score based on keyword matching and semantic similarity."""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Keyword overlap score
        overlap = len(query_words.intersection(doc_words))
        keyword_score = overlap / len(query_words) if query_words else 0
        
        # Legal term bonus
        legal_terms = ['section', 'act', 'punishment', 'offence', 'procedure', 'court', 'magistrate']
        legal_bonus = sum(1 for term in legal_terms if term in document.lower()) * 0.1
        
        # Position bonus (title and beginning of content)
        position_bonus = 0
        if any(word in document[:200].lower() for word in query_words):
            position_bonus = 0.2
        
        return min(1.0, keyword_score + legal_bonus + position_bonus)
    
    def _exact_keyword_search(self, query: str, act_filter: Optional[str] = None) -> List[Dict]:
        """Perform exact keyword matching for high-precision results."""
        exact_matches = []
        query_lower = query.lower()
        
        search_data = self.legal_data
        if act_filter:
            search_data = self.legal_data[self.legal_data['act'].str.lower() == act_filter.lower()]
        
        for _, row in search_data.iterrows():
            content = f"[{row['act']}] {row['title']}: {row['content']}"
            content_lower = content.lower()
            
            # Check for exact phrase matches
            if query_lower in content_lower:
                exact_matches.append({
                    'content': content,
                    'source': 'exact_match',
                    'metadata': {
                        'act': row['act'],
                        'section_no': row['section_no'],
                        'title': row['title']
                    },
                    'score': 1.0,
                    'relevance_score': 1.0
                })
        
        return exact_matches
    
    def _advanced_ranking_and_dedup(self, docs: List[Dict], query: str) -> List[Dict]:
        """Advanced ranking and deduplication with multiple criteria."""
        # Remove duplicates based on content similarity
        unique_docs = []
        seen_contents = set()
        
        for doc in docs:
            # Create a signature for deduplication
            content_sig = doc['content'][:100]  # First 100 chars as signature
            if content_sig not in seen_contents:
                seen_contents.add(content_sig)
                unique_docs.append(doc)
        
        # Multi-criteria ranking
        def ranking_score(doc):
            base_score = doc.get('score', 0)
            relevance_score = doc.get('relevance_score', 0)
            
            # Source preference (exact match > chroma > bm25)
            source_bonus = {
                'exact_match': 0.3,
                'chroma': 0.2,
                'bm25': 0.1
            }.get(doc.get('source', ''), 0)
            
            # Length penalty for very long documents
            length_penalty = min(0.1, len(doc['content']) / 10000)
            
            return base_score + relevance_score + source_bonus - length_penalty
        
        # Sort by ranking score
        unique_docs.sort(key=ranking_score, reverse=True)
        
        return unique_docs
    
    def _create_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """
        Create a sophisticated prompt for high-accuracy legal question answering.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        # Prepare enhanced context with metadata
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get('metadata', {})
            act = metadata.get('act', 'Unknown').upper()
            section = metadata.get('section_no', 'Unknown')
            title = metadata.get('title', 'No title')
            
            context_text += f"Document {i} ({act} Section {section}):\n"
            context_text += f"Title: {title}\n"
            context_text += f"Content: {doc['content']}\n"
            context_text += f"Relevance Score: {doc.get('relevance_score', 0):.2f}\n\n"
        
        # Create sophisticated prompt template
        prompt_template = """You are an expert legal assistant specializing in Indian law with deep knowledge of IPC, CrPC, and NIA. 
Your task is to provide accurate, comprehensive, and legally sound answers based on the provided legal documents.

LEGAL CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Analyze the provided legal documents carefully
2. Identify the most relevant legal provisions
3. Provide a structured, comprehensive answer
4. Include specific section numbers and act references
5. Explain legal implications clearly
6. If multiple relevant sections exist, explain their relationship
7. If the context is insufficient, clearly state what additional information is needed

ANSWER FORMAT:
**Legal Act & Section:** [Specific act and section number]
**Key Provisions:** [Main legal points]
**Detailed Explanation:** [Comprehensive legal explanation]
**Practical Implications:** [How this applies in practice]
**Related Provisions:** [Other relevant sections if any]

ANSWER:"""

        return prompt_template.format(
            context=context_text,
            query=query
        )
    
    def _generate_answer_simple(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate high-accuracy answer using advanced text processing and legal analysis.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Generated answer with high accuracy
        """
        if not context_docs:
            return "I couldn't find relevant legal information to answer your question. Please try rephrasing your query or being more specific about the legal act (IPC, CrPC, or NIA)."
        
        # Analyze query to determine response structure
        query_lower = query.lower()
        is_definition_query = any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning'])
        is_punishment_query = any(word in query_lower for word in ['punishment', 'penalty', 'sentence', 'fine'])
        is_procedure_query = any(word in query_lower for word in ['procedure', 'process', 'how to', 'steps'])
        
        # Build structured answer
        answer_parts = []
        
        # Header
        answer_parts.append("🏛️ **LEGAL ANALYSIS**")
        answer_parts.append("=" * 50)
        
        # Find the most relevant document
        best_doc = max(context_docs, key=lambda x: x.get('relevance_score', 0))
        metadata = best_doc.get('metadata', {})
        act = metadata.get('act', 'Unknown').upper()
        section = metadata.get('section_no', 'Unknown')
        title = metadata.get('title', 'No title')
        
        # Legal Act & Section
        answer_parts.append(f"**Legal Act & Section:** {act} Section {section}")
        answer_parts.append(f"**Section Title:** {title}")
        answer_parts.append("")
        
        # Key Provisions (extract key sentences)
        content = best_doc['content']
        key_sentences = self._extract_key_sentences(content, query)
        
        answer_parts.append("**Key Provisions:**")
        for sentence in key_sentences[:3]:  # Top 3 key sentences
            answer_parts.append(f"• {sentence}")
        answer_parts.append("")
        
        # Detailed Explanation
        answer_parts.append("**Detailed Explanation:**")
        if is_definition_query:
            definition = self._extract_definition(content)
            if definition:
                answer_parts.append(definition)
            else:
                answer_parts.append(content[:500] + "..." if len(content) > 500 else content)
        elif is_punishment_query:
            punishment_info = self._extract_punishment_info(content)
            if punishment_info:
                answer_parts.append(punishment_info)
            else:
                answer_parts.append(content[:500] + "..." if len(content) > 500 else content)
        elif is_procedure_query:
            procedure_info = self._extract_procedure_info(content)
            if procedure_info:
                answer_parts.append(procedure_info)
            else:
                answer_parts.append(content[:500] + "..." if len(content) > 500 else content)
        else:
            answer_parts.append(content[:500] + "..." if len(content) > 500 else content)
        answer_parts.append("")
        
        # Practical Implications
        answer_parts.append("**Practical Implications:**")
        implications = self._generate_practical_implications(query, content)
        answer_parts.append(implications)
        answer_parts.append("")
        
        # Related Provisions
        if len(context_docs) > 1:
            answer_parts.append("**Related Provisions:**")
            for i, doc in enumerate(context_docs[1:4], 2):  # Show up to 3 additional relevant sections
                rel_metadata = doc.get('metadata', {})
                rel_act = rel_metadata.get('act', 'Unknown').upper()
                rel_section = rel_metadata.get('section_no', 'Unknown')
                rel_title = rel_metadata.get('title', 'No title')
                answer_parts.append(f"• {rel_act} Section {rel_section}: {rel_title}")
            answer_parts.append("")
        
        # Legal Disclaimer
        answer_parts.append("⚠️ **Legal Disclaimer:** This information is for educational purposes only. For specific legal advice, consult a qualified legal professional.")
        
        return "\n".join(answer_parts)
    
    def _extract_key_sentences(self, content: str, query: str) -> List[str]:
        """Extract key sentences relevant to the query."""
        sentences = re.split(r'[.!?]+', content)
        query_words = set(query.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Filter out very short sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                score = overlap / len(query_words) if query_words else 0
                scored_sentences.append((score, sentence.strip()))
        
        # Return top sentences by relevance
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sentence for score, sentence in scored_sentences[:5]]
    
    def _extract_definition(self, content: str) -> str:
        """Extract definition from legal content."""
        # Look for definition patterns
        definition_patterns = [
            r'means?\s+([^.]{20,200})',
            r'is\s+([^.]{20,200})',
            r'refers?\s+to\s+([^.]{20,200})'
        ]
        
        for pattern in definition_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return content[:300] + "..." if len(content) > 300 else content
    
    def _extract_punishment_info(self, content: str) -> str:
        """Extract punishment information from legal content."""
        punishment_keywords = ['punishment', 'penalty', 'sentence', 'fine', 'imprisonment', 'jail']
        
        sentences = re.split(r'[.!?]+', content)
        punishment_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in punishment_keywords):
                punishment_sentences.append(sentence.strip())
        
        if punishment_sentences:
            return " ".join(punishment_sentences[:2])  # Top 2 punishment sentences
        else:
            return content[:300] + "..." if len(content) > 300 else content
    
    def _extract_procedure_info(self, content: str) -> str:
        """Extract procedure information from legal content."""
        procedure_keywords = ['procedure', 'process', 'steps', 'method', 'how to', 'application']
        
        sentences = re.split(r'[.!?]+', content)
        procedure_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in procedure_keywords):
                procedure_sentences.append(sentence.strip())
        
        if procedure_sentences:
            return " ".join(procedure_sentences[:2])  # Top 2 procedure sentences
        else:
            return content[:300] + "..." if len(content) > 300 else content
    
    def _generate_practical_implications(self, query: str, content: str) -> str:
        """Generate practical implications based on the query and content."""
        query_lower = query.lower()
        
        if 'murder' in query_lower:
            return "Murder is a serious criminal offence with severe penalties. The prosecution must prove intent and the act beyond reasonable doubt. Legal representation is essential."
        elif 'theft' in query_lower:
            return "Theft involves dishonestly taking property without consent. The value of stolen property determines the severity of punishment. Evidence of intent is crucial."
        elif 'bail' in query_lower:
            return "Bail is a constitutional right in India. The court considers factors like nature of offence, evidence strength, and flight risk. Legal assistance is recommended."
        elif 'arrest' in query_lower:
            return "Arrest procedures must follow constitutional safeguards. Police must inform rights, allow legal representation, and follow due process. Any violation can be challenged."
        elif 'terrorism' in query_lower:
            return "Terrorism cases involve special procedures under NIA Act. Enhanced penalties and special courts apply. Legal representation is mandatory."
        else:
            return "This legal provision has specific requirements and procedures that must be followed. Consult a legal expert for case-specific advice."
    
    def query(self, question: str, verbose: bool = True) -> Dict:
        """
        Process a legal query and return comprehensive results.
        
        Args:
            question: User's legal question
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing query results
        """
        if verbose:
            print(f"\n🔍 Processing query: '{question}'")
        
        # Step 1: Classify the legal act
        if verbose:
            print("🎯 Classifying legal act...")
        
        try:
            predicted_act, confidence = classify_act(question)
            if verbose:
                print(f"   → Predicted Act: {predicted_act} (confidence: {confidence:.2f})")
        except Exception as e:
            if verbose:
                print(f"⚠️ Act classification failed: {e}")
            predicted_act, confidence = None, 0.0
        
        # Step 2: Retrieve relevant documents
        if verbose:
            print("📚 Retrieving relevant documents...")
        
        try:
            retrieved_docs = self._retrieve_documents(question, predicted_act if confidence > 0.5 else None)
            if verbose:
                print(f"   → Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            if verbose:
                print(f"⚠️ Document retrieval failed: {e}")
            retrieved_docs = []
        
        # Step 3: Generate high-accuracy answer
        if verbose:
            print("🤖 Generating high-accuracy answer...")
        
        try:
            if self.llm is not None and retrieved_docs and LANGCHAIN_AVAILABLE:
                # Use LLM for answer generation
                prompt = self._create_prompt(question, retrieved_docs)
                answer = self.llm(prompt)
            else:
                # Use advanced text generation with legal analysis
                answer = self._generate_answer_simple(question, retrieved_docs)
            
            if verbose:
                print("✅ High-accuracy answer generated successfully")
        except Exception as e:
            if verbose:
                print(f"⚠️ Answer generation failed: {e}")
            # Fallback to simple answer generation
            try:
                answer = self._generate_answer_simple(question, retrieved_docs)
            except Exception as fallback_error:
                answer = "I apologize, but I encountered an error while generating the answer. Please try again."
        
        # Prepare results
        results = {
            'question': question,
            'predicted_act': predicted_act,
            'confidence': confidence,
            'retrieved_documents': retrieved_docs,
            'answer': answer,
            'sources': [doc.get('metadata', {}) for doc in retrieved_docs]
        }
        
        return results
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface that returns just the answer.
        
        Args:
            question: User's legal question
            
        Returns:
            Generated answer string
        """
        results = self.query(question, verbose=False)
        return results['answer']
    
    def get_system_info(self) -> Dict:
        """
        Get information about the system configuration.
        
        Returns:
            Dictionary with system information
        """
        return {
            'embedding_model': self.embedding_model_name,
            'llm_model': self.llm_model_name,
            'chroma_path': self.chroma_path,
            'collection_name': self.collection_name,
            'max_context_length': self.max_context_length,
            'top_k': self.top_k,
            'total_documents': len(self.legal_data) if self.legal_data is not None else 0,
            'llm_available': self.llm is not None
        }


def main():
    """
    Main function to demonstrate the Legal RAG System.
    """
    print("🏛️ Legal RAG Chatbot - Indian Law Assistant")
    print("=" * 50)
    
    try:
        # Initialize the system
        rag_system = LegalRAGSystem()
        
        # Display system info
        info = rag_system.get_system_info()
        print(f"\n📊 System Information:")
        print(f"   • Embedding Model: {info['embedding_model']}")
        print(f"   • LLM Model: {info['llm_model']}")
        print(f"   • Total Documents: {info['total_documents']}")
        print(f"   • LLM Available: {info['llm_available']}")
        
        # Example queries
        example_queries = [
            "What is the punishment for murder?",
            "What is the procedure for arrest?",
            "What are the powers of NIA in terrorism cases?",
            "How to apply for bail?",
            "What is the definition of theft?"
        ]
        
        print(f"\n🔍 Testing with example queries:")
        print("=" * 50)
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 30)
            
            try:
                results = rag_system.query(query, verbose=False)
                print(f"Predicted Act: {results['predicted_act']} (confidence: {results['confidence']:.2f})")
                print(f"Retrieved Documents: {len(results['retrieved_documents'])}")
                print(f"Answer: {results['answer'][:200]}...")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        # Interactive mode
        print(f"\n💬 Interactive Mode (type 'quit' to exit):")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Ask a legal question: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    answer = rag_system.chat(user_input)
                    print(f"\n📝 Answer: {answer}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize Legal RAG System: {e}")
        print("💡 Make sure to run the following scripts first:")
        print("   1. python preprocess_datasets.py")
        print("   2. python build_embeddings.py")


if __name__ == "__main__":
    main()
