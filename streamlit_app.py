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
                    max_tokens=1000,
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
        page_title="RAG Chat Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Modern, clean CSS
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Chat messages styling */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 12px 0 12px auto;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        font-size: 15px;
        line-height: 1.6;
    }
    
    .assistant-message {
        background: #f8fafc;
        color: #1e293b;
        padding: 16px 20px;
        border-radius: 20px 20px 20px 4px;
        margin: 12px auto 12px 0;
        max-width: 80%;
        border: 1px solid #e2e8f0;
        word-wrap: break-word;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        font-size: 15px;
        line-height: 1.6;
    }
    
    .message-timestamp {
        font-size: 11px;
        opacity: 0.6;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Chat container */
    .chat-messages {
        height: calc(100vh - 300px);
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Scrollbar styling */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 80px 20px;
        color: #64748b;
    }
    
    .welcome-container h2 {
        color: #334155;
        font-size: 28px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    
    .welcome-container p {
        font-size: 16px;
        line-height: 1.6;
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Source badges */
    .source-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .github-tag {
        background: #dcfce7;
        color: #166534;
    }
    
    .slack-tag {
        background: #fce7f3;
        color: #9f1239;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-card h3 {
        font-size: 14px;
        font-weight: 600;
        color: #475569;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 14px;
    }
    
    .stat-item:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        color: #64748b;
        font-weight: 500;
    }
    
    .stat-value {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        font-size: 14px;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 14px 18px;
        font-size: 15px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        font-weight: 500;
        font-size: 14px;
    }
    
    /* Remove extra padding */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Header styling */
    h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 32px;
        margin-bottom: 8px;
    }
    
    /* Example questions */
    .example-question {
        background: white;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e2e8f0;
        font-size: 14px;
        color: #475569;
    }
    
    .example-question:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        transform: translateX(4px);
    }
    
    /* Processing indicator */
    .processing-indicator {
        color: #6366f1;
        font-size: 14px;
        font-weight: 500;
        padding: 12px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check API keys
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or os.getenv('PINECONE_API_KEY')
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv('ANTHROPIC_API_KEY')
    
    if not pinecone_key or not anthropic_key:
        st.error("⚠️ Please set your API keys in Streamlit secrets or environment variables")
        st.stop()
    
    # Initialize RAG system
    @st.cache_resource
    def init_rag_system():
        return LangGraphRAGSystem(pinecone_index_name="turbo-rag-index")
    
    try:
        rag_system = init_rag_system()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {e}")
        st.stop()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 💬 RAG Assistant")
        st.markdown("---")
        
        # Settings card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("<h3>⚙️ Settings</h3>", unsafe_allow_html=True)
        
        source_filter = st.selectbox(
            "Search source",
            options=["both", "github", "slack"],
            format_func=lambda x: {
                "both": "🔍 All Sources",
                "github": "💻 GitHub Only", 
                "slack": "💬 Slack Only"
            }[x],
            label_visibility="visible"
        )
        
        top_k = st.slider(
            "Results to retrieve",
            min_value=3,
            max_value=10,
            value=5
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stats card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("<h3>📊 Statistics</h3>", unsafe_allow_html=True)
        
        stats = rag_system.get_index_stats()
        if stats:
            st.markdown(f"""
            <div class="stat-item">
                <span class="stat-label">Documents</span>
                <span class="stat-value">{stats.get('total_vectors', 0):,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Index Usage</span>
                <span class="stat-value">{stats.get('index_fullness', 0):.1%}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Messages</span>
                <span class="stat-value">{len([m for m in st.session_state.chat_history if m.role == 'user'])}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actions card
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("<h3>🎯 Actions</h3>", unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Export Chat", use_container_width=True, disabled=len(st.session_state.chat_history) == 0):
            chat_export = []
            for msg in st.session_state.chat_history:
                chat_export.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
            
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(chat_export, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Example questions
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("<h3>💡 Try asking</h3>", unsafe_allow_html=True)
        
        examples = [
            "How does authentication work?",
            "What deployment issues were discussed?",
            "Show me the database schema",
            "Summarize recent team decisions"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat area
    st.title("💬 Chat with Your Knowledge Base")
    st.markdown("Ask questions about your GitHub repository and Slack conversations")
    st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
            
            for msg in st.session_state.chat_history:
                time_str = msg.timestamp.strftime("%I:%M %p")
                
                if msg.role == "user":
                    st.markdown(f'''
                    <div class="user-message">
                        {msg.content}
                        <div class="message-timestamp">{time_str}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="assistant-message">
                        {msg.content}
                        <div class="message-timestamp">{time_str}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Show sources
                    if msg.sources:
                        with st.expander(f"📚 {len(msg.sources)} sources used", expanded=False):
                            for idx, source in enumerate(msg.sources, 1):
                                source_type = source['source_type']
                                tag_class = "github-tag" if source_type == "github" else "slack-tag"
                                
                                st.markdown(f'<span class="source-tag {tag_class}">{source_type}</span>', unsafe_allow_html=True)
                                
                                if source_type == 'github':
                                    st.markdown(f"**File:** `{source['file_path']}`")
                                elif source_type == 'slack':
                                    st.markdown(f"**Channel:** #{source['channel']} • **User:** {source['user']}")
                                
                                preview = source['content'][:250] + "..." if len(source['content']) > 250 else source['content']
                                st.markdown(f"```\n{preview}\n```")
                                
                                if idx < len(msg.sources):
                                    st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="welcome-container">
                <h2>👋 Welcome!</h2>
                <p>Start a conversation by typing your question below. I can help you explore your GitHub repository and Slack conversations.</p>
                <p style="margin-top: 20px; color: #94a3b8;">Try asking about code implementations, team discussions, or project decisions.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Input area at bottom
    st.markdown("---")
    
    # Handle pending question from sidebar
    default_question = ""
    if 'pending_question' in st.session_state:
        default_question = st.session_state.pending_question
        del st.session_state.pending_question
    
    # Create input form
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Message",
            value=default_question,
            placeholder="Ask me anything about your codebase or team conversations...",
            label_visibility="collapsed",
            disabled=st.session_state.processing,
            key="main_input"
        )
    
    with col2:
        send_clicked = st.button(
            "Send",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing or not question.strip()
        )
    
    # Process the question
    if send_clicked and question.strip() and not st.session_state.processing:
        st.session_state.processing = True
        
        # Add user message
        user_message = ChatMessage(
            role="user",
            content=question,
            timestamp=datetime.now()
        )
        st.session_state.chat_history.append(user_message)
        
        # Show processing indicator
        with st.spinner("🤔 Thinking..."):
            result = rag_system.chat(
                question=question,
                source_filter=source_filter,
                top_k=top_k,
                conversation_history=st.session_state.chat_history[:-1]
            )
            
            # Add assistant message
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
        
        st.session_state.processing = False
        st.rerun()

if __name__ == "__main__":
    main()
