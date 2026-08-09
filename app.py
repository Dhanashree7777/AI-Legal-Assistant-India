import streamlit as st
import requests

# Set the design of the webpage
st.set_page_config(page_title="Indian Legal AI", page_icon="🏛️", layout="centered")

# Your FastAPI URLs
API_URL = "http://127.0.0.1:8000/execute-workflow"
CLEAR_MEMORY_URL = "http://127.0.0.1:8000/clear-memory"

# Title of the app
st.title("⚖️ AI Legal Assistant for Indian Laws")
st.markdown("Ask me anything about the Constitution, BNS, or other Indian laws.")

# Sidebar with the "Clear Chat" button
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Chat History"):
        try:
            requests.post(CLEAR_MEMORY_URL) # Tells FastAPI to wipe the memory
            st.session_state.messages = []  # Wipes the screen memory
            st.success("Memory cleared! Ready for a new case.")
        except:
            st.error("Error connecting to backend.")

# Initialize the chat history on the screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box
if prompt := st.chat_input("E.g., What is Section 318 of the BNS?"):
    
    # 1. Print the user's question on the screen
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Show a loading spinner while the AI thinks
    with st.spinner("Searching the Vector Database and Analyzing Laws..."):
        try:
            # Send the question to your FastAPI backend
            response = requests.post(f"{API_URL}?query={prompt}")
            
            if response.status_code == 200:
                bot_reply = response.json() # Extract the text from the response
            else:
                bot_reply = "⚠️ Sorry, the backend encountered an error."
        except Exception as e:
            bot_reply = "⚠️ Error: Could not connect to the FastAPI server. Is it running?"

    # 3. Print the AI's answer on the screen
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})