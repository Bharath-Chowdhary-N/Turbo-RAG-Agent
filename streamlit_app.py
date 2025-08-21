import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import anthropic
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

# Load environment variables for local development only
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Handle Pinecone import with compatibility
try:
    from pinecone import Pinecone
except ImportError:
    st.error("Please update your requirements.txt to use 'pinecone' instead of 'pinecone-client'")
    st.stop()

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None

class ConversationState(TypedDict):
    """State for the LangGraph conversation workflow"""
    messages: List[Dict[str, Any]]
    current_question: str
    search_results: List[Dict[str, Any]]
    context: str
    response: str
    source_filter: str
    top_k: int
    conversation_history: List[ChatMessage]

class LangGraphRAGSystem:
    def __init__(self, pinecone_index_name: str = "turbo-rag-index"):
        self.pinecone_index_name = pinecone_index_name
        
        # Initialize Pinecone
        try:
            pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                st.error("PINECONE_API_KEY not found in secrets or environment variables")
                st.stop()
                
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = self.pinecone_client.Index(pinecone_index_name)
        except Exception as e:
            st.error(f"Failed to connect to Pinecone: {e}")
            st.stop()
        
        # Initialize embedding model
        @st.cache_resource
        def load_embedder():
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        self.embedder = load_embedder()
        
        # Initialize Anthropic Claude
        try:
            anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                st.error("ANTHROPIC_API_KEY not found in secrets or environment variables")
                st.stop()
                
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Claude: {e}")
            st.stop()
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph conversation workflow"""
        
        def search_step(state: ConversationState) -> ConversationState:
            """Search for relevant content"""
            query = state["current_question"]
            source_filter = state.get("source_filter", "both")
            top_k = state.get("top_k", 5)
            
            search_results = self._search_relevant_content(query, top_k, source_filter)
            state["search_results"] = search_results
            return state
        
        def context_step(state: ConversationState) -> ConversationState:
            """Generate context from search results and conversation history"""
            search_results = state["search_results"]
            conversation_history = state.get("conversation_history", [])
            
            # Generate context from search results
            search_context = self._generate_context(search_results)
            
            # Generate conversation context from recent history
            conversation_context = self._generate_conversation_context(conversation_history)
            
            # Combine contexts
            full_context = f"{conversation_context}\n\n{search_context}"
            state["context"] = full_context
            return state
        
        def response_step(state: ConversationState) -> ConversationState:
            """Generate response using Claude"""
            question = state["current_question"]
            context = state["context"]
            source_filter = state.get("source_filter", "both")
            conversation_history = state.get("conversation_history", [])
            
            # Create conversational prompt
            prompt = self._create_conversational_prompt(question, context, source_filter, conversation_history)
            
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                state["response"] = response.content[0].text
            except Exception as e:
                state["response"] = f"Error generating response: {str(e)}"
            
            return state
        
        # Build the graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("search", search_step)
        workflow.add_node("context", context_step)
        workflow.add_node("response", response_step)
        
        # Add edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "context")
        workflow.add_edge("context", "response")
        workflow.add_edge("response", END)
        
        return workflow.compile()
    
    def _search_relevant_content(self, query: str, top_k: int = 5, source_filter: str = None) -> List[Dict[str, Any]]:
        """Search for relevant content in Pinecone"""
        try:
            query_embedding = self.embedder.encode([query])[0].tolist()
            
            filter_dict = None
            if source_filter and source_filter != "both":
                filter_dict = {"source_type": source_filter}
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            search_results = []
            for match in results['matches']:
                search_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'content': match['metadata'].get('content_preview', ''),
                    'source_type': match['metadata'].get('source_type', 'unknown'),
                    'file_path': match['metadata'].get('file_path', ''),
                    'channel': match['metadata'].get('channel', ''),
                    'user': match['metadata'].get('user', ''),
                    'timestamp': match['metadata'].get('timestamp', ''),
                    'metadata': match['metadata']
                })
            
            return search_results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def _generate_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Generate context from search results"""
        if not search_results:
            return "No relevant content found."
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            source_type = result['source_type']
            content = result['content']
            
            if source_type == 'github':
                file_path = result['file_path']
                context_part = f"[GitHub Code - {file_path}]\n{content}\n"
            elif source_type == 'slack':
                channel = result['channel']
                user = result['user']
                timestamp = result['timestamp']
                context_part = f"[Slack - #{channel} - {user} at {timestamp}]\n{content}\n"
            else:
                context_part = f"[Source: {source_type}]\n{content}\n"
            
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _generate_conversation_context(self, conversation_history: List[ChatMessage]) -> str:
        """Generate context from recent conversation history"""
        if not conversation_history:
            return ""
        
        # Include last 5 messages for context
        recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        context_parts = ["RECENT CONVERSATION HISTORY:"]
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def _create_conversational_prompt(self, question: str, context: str, source_filter: str, conversation_history: List[ChatMessage]) -> str:
        """Create a conversational prompt for Claude"""
        source_description = {
            "github": "GitHub repository code and documentation",
            "slack": "Slack team conversations and discussions", 
            "both": "both GitHub repository and Slack conversations"
        }
        
        has_history = len(conversation_history) > 0
        
        prompt = f"""You are an expert assistant with access to {source_description.get(source_filter, 'various sources')}. 
You are having a conversation with a user about their project.

{'This is a continuation of an ongoing conversation. Please maintain context and refer back to previous topics when relevant.' if has_history else 'This is the start of a new conversation.'}

RELEVANT INFORMATION FROM SEARCH:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, helpful response that builds on the conversation context
2. If this relates to previous questions, acknowledge that connection
3. For code questions, explain functionality and provide examples
4. For team discussions, summarize key points and decisions
5. Use a conversational, natural tone - you're having a dialogue
6. If information is insufficient, clearly state what's missing
7. Format code snippets with proper markdown
8. Keep responses focused but comprehensive

RESPONSE:"""
        
        return prompt
    
    def chat(self, question: str, source_filter: str = "both", top_k: int = 5, conversation_history: List[ChatMessage] = None) -> Dict[str, Any]:
        """Process a chat message through the LangGraph workflow"""
        start_time = time.time()
        
        if conversation_history is None:
            conversation_history = []
        
        # Create initial state
        initial_state: ConversationState = {
            "messages": [],
            "current_question": question,
            "search_results": [],
            "context": "",
            "response": "",
            "source_filter": source_filter,
            "top_k": top_k,
            "conversation_history": conversation_history
        }
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            total_time = time.time() - start_time
            
            return {
                'answer': final_state["response"],
                'sources': final_state["search_results"],
                'processing_time': total_time,
                'context_used': len(final_state["search_results"])
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing your question: {str(e)}",
                'sources': [],
                'processing_time': time.time() - start_time,
                'context_used': 0
            }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'namespaces': stats.get('namespaces', {})
            }
        except Exception as e:
            st.error(f"Error getting index stats: {e}")
            return {}

def main():
    # Page configuration
    st.set_page_config(
        page_title="LangGraph RAG Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    .assistant-message {
        background-color: white;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #ddd;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .github-badge {
        background-color: #28a745;
        color: white;
    }
    .slack-badge {
        background-color: #4a154b;
        color: white;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LangGraph RAG Chat Assistant</h1>
        <p>Interactive chat with your GitHub repository and Slack conversations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API keys
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or os.getenv('PINECONE_API_KEY')
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv('ANTHROPIC_API_KEY')
    
    if not pinecone_key or not anthropic_key:
        st.error("Please set your API keys in Streamlit secrets or environment variables")
        st.stop()
    
    # Initialize RAG system
    @st.cache_resource
    def init_rag_system():
        return LangGraphRAGSystem(pinecone_index_name="turbo-rag-index")
    
    try:
        rag_system = init_rag_system()
        st.success("Connected to Pinecone and Claude successfully!")
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Source filter
        source_filter = st.selectbox(
            "Search in:",
            options=["both", "github", "slack"],
            format_func=lambda x: {
                "both": "Both GitHub & Slack",
                "github": "GitHub Repository Only", 
                "slack": "Slack Messages Only"
            }[x]
        )
        
        # Number of results
        top_k = st.slider(
            "Number of results to retrieve:",
            min_value=3,
            max_value=10,
            value=5
        )
        
        # Chat controls
        st.subheader("Chat Controls")
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("Export Chat", type="secondary"):
            if st.session_state.chat_history:
                chat_export = []
                for msg in st.session_state.chat_history:
                    chat_export.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    })
                
                st.download_button(
                    label="Download Chat History",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Index statistics
        st.subheader("Database Stats")
        with st.spinner("Loading stats..."):
            stats = rag_system.get_index_stats()
            
        if stats:
            st.markdown(f"""
            <div class="stats-container">
                <strong>Total Documents:</strong> {stats.get('total_vectors', 0):,}<br>
                <strong>Index Fullness:</strong> {stats.get('index_fullness', 0):.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Example questions
        st.subheader("Example Questions")
        example_questions = [
            "How does authentication work?",
            "What did the team discuss about the API?",
            "Show me the main functions",
            "What deployment issues were mentioned?",
            "How is the database configured?",
            "What are recent architectural decisions?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}", use_container_width=True):
                st.session_state.pending_question = question
    
    # Main chat interface
    st.subheader("Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for msg in st.session_state.chat_history:
                if msg.role == "user":
                    st.markdown(f'<div class="user-message">{msg.content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message">{msg.content}</div>', unsafe_allow_html=True)
                    
                    # Show sources if available
                    if msg.sources:
                        with st.expander(f"Sources ({len(msg.sources)} found)", expanded=False):
                            for i, source in enumerate(msg.sources, 1):
                                source_type = source['source_type']
                                badge_class = "github-badge" if source_type == "github" else "slack-badge"
                                
                                st.markdown(f'<span class="source-badge {badge_class}">{source_type.upper()}</span>', unsafe_allow_html=True)
                                
                                if source_type == 'github':
                                    st.text(f"File: {source['file_path']}")
                                elif source_type == 'slack':
                                    st.text(f"#{source['channel']} - {source['user']}")
                                
                                st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                                st.divider()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Start a conversation by asking a question below!")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Handle pending question from sidebar
        default_question = ""
        if 'pending_question' in st.session_state:
            default_question = st.session_state.pending_question
            del st.session_state.pending_question
        
        question = st.text_input(
            "Ask a question:",
            value=default_question,
            placeholder="e.g., How does the authentication system work?",
            key="question_input"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process question
    if (send_button or question) and question.strip():
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=question,
            timestamp=datetime.now()
        )
        st.session_state.chat_history.append(user_message)
        
        # Get response from RAG system
        with st.spinner("Thinking..."):
            result = rag_system.chat(
                question=question,
                source_filter=source_filter,
                top_k=top_k,
                conversation_history=st.session_state.chat_history[:-1]  # Exclude current user message
            )
            
            # Add assistant message to history
            assistant_message = ChatMessage(
                role="assistant",
                content=result['answer'],
                timestamp=datetime.now(),
                sources=result['sources'],
                metadata={
                    'processing_time': result['processing_time'],
                    'context_used': result['context_used']
                }
            )
            st.session_state.chat_history.append(assistant_message)
        
        # Clear input and rerun to show new messages
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Chat History: {len([msg for msg in st.session_state.chat_history if msg.role == 'user'])} messages")

if __name__ == "__main__":
    main()
