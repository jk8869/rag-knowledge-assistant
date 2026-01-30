import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="My Knowledge Assistant", page_icon="🤖")

st.title("🤖 Internal Knowledge Assistant")

# --- 1. File Upload Section ---
st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Button to trigger upload
    if st.sidebar.button("Process Document"):
        with st.spinner("Uploading & Chunking..."):
            # Send file to FastAPI
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post("http://127.0.0.1:8000/upload", files=files)
            
            if response.status_code == 200:
                st.sidebar.success(f"✅ Processed! {response.json().get('chunks_added')} chunks added.")
            else:
                st.sidebar.error("❌ Upload failed.")

# --- 2. Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add user message to UI immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The payload must match your QueryRequest model in main.py
                payload = {"question": prompt} 
                response = requests.post("http://127.0.0.1:8000/ask", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer")
                    sources = response.json().get("sources", [])
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # (Optional) Display Sources nicely
                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources:
                                st.text(source[:300] + "...") # Show first 300 chars
                                st.divider()
                                
                    # Save assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                else:
                    st.error("⚠️ Backend Error: Could not get answer.")
                    
            except Exception as e:
                st.error(f"❌ Connection Error: {e}")