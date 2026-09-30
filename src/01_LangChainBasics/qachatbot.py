"""
Simple LangChain Streamlit App With Groq
"""

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Page Config
st.set_page_config(
    page_title="LangChain Chat with Groq",
    page_icon="🤖",
)

st.title("LangChain Chat with Groq")
st.markdown("""
This is a simple LangChain Streamlit app that uses Groq as the LLM.
""")

with st.sidebar:
    st.header("Settings")

    ## Api Key
    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free Groq API key from console.groq.com",
    )

    ## Model Selection
    model_name = st.selectbox(
        "Model",
        options=["qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        index=0,
    )

    ## Clear Chat Button
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize LLM
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        return None

    chat_model = init_chat_model(model="groq:" + model_name, api_key=api_key, temperature=0.7, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant powered by groq. Answer questions clearly and concisely."),
        ("user", "{question}"),
    ])

    # Create chain
    chain = prompt | chat_model | StrOutputParser()
    return chain

# Get the chain
chain = get_chain(api_key, model_name)
if not chain:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.markdown("## Please enter your Groq API key in the sidebar to start chatting.")
else:
    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({ "question": question }):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                st.session_state["messages"].append({"role": "assistant", "content": full_response})

            except Exception as e:
                print(e)
                st.error(f"Error: {str(e)}")