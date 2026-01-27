import streamlit as st

from chat_utils import hide_sidebar_nav


def hide_sidebar():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Climate Change AI Assistant",
    page_icon="💬",
    layout="wide",
)
hide_sidebar_nav()
hide_sidebar()

st.title("Climate Change AI Assistant")
st.write(
    "Welcome to the Climate Change AI Assistant. "
    "If you reached this page directly, please return to the provided link."
)
