import streamlit as st
import json
from agent import SupportAgent

# Page configuration
st.set_page_config(
    page_title="AI Support Assistant 🤖",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = SupportAgent()
    st.session_state.messages = []

# UI
st.title("🤖 AI Support Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Analytics")
    analytics = st.session_state.agent.get_analytics()
    
    if analytics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", analytics['total_queries'])
        with col2:
            st.metric("FAQ Resolution", f"{analytics['faq_resolution_rate']}%")
        with col3:
            st.metric("Escalation Rate", f"{analytics['escalation_rate']}%")
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
st.subheader("💬 Chat with Support")

# Display conversation
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.chat_message("user").write(msg['content'])
    else:
        with st.chat_message("assistant"):
            st.write(msg['content']['message'])
            
            # Show metadata
            with st.expander("📋 Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Confidence**: {msg['content']['confidence_score']}")
                with col2:
                    st.write(f"**Type**: {msg['content']['response_type']}")
                with col3:
                    st.write(f"**Category**: {msg['content']['category']}")
                
                if msg['content']['sources']:
                    st.write("**Sources:**")
                    for source in msg['content']['sources']:
                        st.write(f"- {source['id']}: {source['title']}")

# Input
user_input = st.chat_input("Type your support question here...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    # Get response
    response = st.session_state.agent.process_query(user_input)
    
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })
    
    st.rerun()

# Footer
st.markdown("---")
st.caption("🚀 Powered by LangChain | Hybrid RAG | AI Support")
