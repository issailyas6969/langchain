import validators,streamlit as st
from langchain.ptompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader



st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="📄",
    layout="centered"
)

st.title("📄 LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize URL")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


pages_template="""Write a summary of the following content in 300 words:
speech:{text}

"""
prompt=PromptTemplate(input_variables=['text'],template=pages_template)
# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

url=st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize the content from yt or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url,it may be a yt video or website url")
    else:
        try:
            with st.spinner("Waiting,,,"):
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,headers=)
                
                docs=loader.load()

                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
