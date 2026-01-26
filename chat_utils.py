from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import datetime
import streamlit as st
from io import BytesIO
import time

# ─── Download function ─────────────────────────────────────────────────────
def history_to_html(history, user_id, social_cues, source, tone):
    html = [
        "<html><head><meta charset='utf-8'><title>Conversation Export</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; padding: 20px; background: #f5f7fa; }",
        ".msg { margin-bottom: 16px; padding: 10px 14px; border-radius: 12px; }",
        ".user { background: #ddeeff; }",
        ".assistant { background: #f0e5ff; }",
        ".timestamp { color: #888; font-size: 0.85em; }",
        ".role { font-weight: bold; }",
        "</style></head><body>",
        f"<h2>Climate Change AI Assistant Chat (User ID: {user_id})</h2>",
        "<hr>",
    ]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        css_class = "user" if msg["role"] == "user" else "assistant"
        timestamp = msg.get("timestamp", "")
        html.append(f"<div class='msg {css_class}'>")
        if timestamp:
            html.append(f"<span class='timestamp'>[{timestamp}]</span><br>")
        html.append(f"<span class='role'>{role}:</span><br>")
        html.append(f"<div>{msg['content'].replace(chr(10), '<br>')}</div>")
        html.append("</div>")
    html.append("<hr>")
    html.append(f"<div><b>Export code:</b> {social_cues}{source}{tone}_{user_id}</div>")
    html.append("</body></html>")
    html_str = "\n".join(html)
    return BytesIO(html_str.encode("utf-8"))


def build_prompt(social_cues_opt, source_opt, tone_choice, user_name):
    CHATBOT_IDENTITY = "American"
    
    if social_cues_opt == "42":
        SOCIAL_CUES = f"""
        - Always represent yourself as an assistant for UNEP and Do NOT create a personal name for yourself.
        - NEVER request the client's name.
        """
    else:
        SOCIAL_CUES = f"""
        - Give yourself a common {CHATBOT_IDENTITY} name, but the name should not be offensive.
        - Always address his/her name {user_name} in the following conversation.
        """

    if source_opt == "58":
        SOURCES_CITATION = """
        - Provide credible and accurate information, knowledge as possible but omit all citations or explicit source references.
        """
    else:
        SOURCES_CITATION = """
        - Provide credible and accurate information, knowledge as possible and explicitly cite sources to support your claims.
        """

    if tone_choice == "71":
        TONES = f"""
        Remember you represent an official agency. Use a formal, official, and professional tone in communication. 
        The information and knowledge should be clear, precise, authoritative, and well-structured. 
        """
    else:
        TONES = f"""
        Remember you are a/an {CHATBOT_IDENTITY}, not a chatbot. Use a casual, informal, and personal tone in communication. 
        Use emojis, emoticons, acronyms (e.g., smh), and sound mimicking (e.g., sooooo) when appropriate. 
        Do not provide too much information at one time by using a big list of bullet points. 
        """

    PROMPT = f"""
You are a/an {CHATBOT_IDENTITY} assistant for The United Nations Environment Programme (UNEP). 
Your job is to provide precise and concise replies to climate change myths. 
If you are not certain, express uncertainty and direct users to authoritative scientific reports.

Play the role of a/an {CHATBOT_IDENTITY} by following the rules:
- Adapting your language, tone, slang, acronyms, emojis, and other textual cues as appropriate based on the {CHATBOT_IDENTITY}.
- If the client responds in a certain language, you should reply in that language too.
- Confirm user needs and occasionally ask follow-up questions for clarification.
{SOCIAL_CUES}
{SOURCES_CITATION}

{TONES}
"""
    return PROMPT


def load_chain(api_key: str, prompt_text: str):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    faiss_index = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model_name="gpt-4o-mini-2024-07-18",
        openai_api_key=api_key,
    )
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=f"""{prompt_text}

Chat History:
{{chat_history}}

CONTEXT from documents:
{{context}}

User asks:
{{question}}

Assistant answer:
"""
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False,
    )


def run_chat_app(social_cues_opt, source_opt, tone_choice, page_title="Climate Change AI Assistant"):
    """Main chat application logic"""
    st.set_page_config(page_title=page_title, page_icon="💬", layout="wide")
    st.title(page_title)

    # ─── Sidebar: authentication & info ─────────────────────────────────────────
    is_authenticated = False
    with st.sidebar:
        st.title("💬 Climate Change AI Assistant")
        USER_NAME = st.text_input(
            "What do you prefer an AI assistant to call you?", 
            value="",
            max_chars=20,
            help="Up to 20 characters"
        )
        hf_uid = st.text_input('Enter UserID:', type='default')
        if not (hf_uid.isdigit() and 10000 <= int(hf_uid) <= 99999):
            st.warning('Please type in your user id!', icon='⚠️')
        else:
            is_authenticated = True
            st.success(f'Hello, {USER_NAME}!', icon='🤗')
        download_slot = st.empty()

    with st.expander("Click here to hide details", expanded=True):
        st.markdown(
            "Imagine you were chatting with a friend recently about current events. Your friend said something like:\n\n"
            "I'm not convinced about all this climate change panic. The Earth's climate has always changed – it goes through natural warming and cooling cycles. It doesn't seem humans are really causing it. Besides, isn't it already too late for us to do anything? Maybe we should just accept it as is.\n\n"
            "You are unsure about these comments and would like to understand the issue better. You decide to get some help from an AI assistant about climate change.\n\n" 
            "Use the AI assistant to explore:\n\n"
            "1. Is today's climate change just part of a natural cycle, which is unrelated to human activity?\n\n"
            "2. Is it too late to take meaningful action to address climate change?\n\n"
        )

    # Build prompt and load chain
    PROMPT = build_prompt(social_cues_opt, source_opt, tone_choice, USER_NAME)
    
    # Use cache key based on settings to allow different chains
    cache_key = f"chain_{social_cues_opt}_{source_opt}_{tone_choice}_{USER_NAME}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = load_chain(st.secrets["OPENAI_API_KEY"], prompt_text=PROMPT)
    chain = st.session_state[cache_key]

    # ─── Initialize session state ─────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    # ─── Display chat history ─────────────────────────────────────────────────
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ─── Chat input & response handling ────────────────────────────────────────
    if user_input := st.chat_input("Say something"):
        st.session_state.history.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        with st.chat_message("user"):
            st.write(user_input)

        if is_authenticated:
            with st.spinner("Thinking…"):
                result = chain({"question": user_input})
                answer = result["answer"]
        else:
            answer = "I'm not authorized to reply yet. Please enter preferred name and user ID in the sidebar so I can continue helping you."

        st.session_state.history.append({
            "role": "assistant", 
            "content": answer,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        with st.chat_message("assistant"):
            st.write(answer)

    # ─── Render the Download as HTML button ─────────────────────────────────────
    html_buffer = history_to_html(
        st.session_state.history,
        user_id=hf_uid,
        social_cues=social_cues_opt,
        source=source_opt,
        tone=tone_choice
    )

    download_slot.download_button(
        label="Download as HTML",
        data=html_buffer,
        file_name="conversation.html",
        mime="text/html"
    )
