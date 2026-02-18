import os
import json
import streamlit as st
import requests

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

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
            response = requests.post(f"{API_URL}/upload", files=files)
            
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

    # --- MISSING PART ADDED HERE ---
    # Create history (exclude the current prompt to avoid duplication in backend)
    history_payload = [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.messages[:-1]
    ]

    payload = {
        "question": prompt,
        "messages": history_payload
    }
    # -------------------------------

    # 2. Call FastAPI Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        spinner = st.spinner("Thinking...")
        spinner.__enter__()

        with requests.post(f"{API_URL}/ask", json=payload, stream=True) as response:
            if response.status_code == 200:
                try:
                    first_token_received = False

                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))

                            if data.get("type") == "meta":
                                sources = data.get("sources", [])

                            elif data.get("type") == "token":
                                if not first_token_received:
                                    spinner.__exit__(None, None, None)
                                    first_token_received = True

                                content = data.get("content", "")
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    if sources:
                        with st.expander("View Sources"):
                            for s in sources:
                                st.write(s)
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                except json.JSONDecodeError:
                    st.error("Error: Failed to decode stream.")
            else:
                st.error(f"Error: {response.status_code}")

                    