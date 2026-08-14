import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="YouTube RAG Chatbot")
st.title("Chat with a YouTube video")

video_id_input = st.text_input("YouTube video ID", value="Gfr50f6ZBvo")
go = st.button("Load video")


@st.cache_resource(show_spinner="Indexing video transcript...")
def build_chain(video_id):
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.fetch(video_id, languages=["en"])
    except (TranscriptsDisabled, NoTranscriptFound):
        st.session_state.load_error = "No English captions available for this video."
        return None
    except Exception as e:
        st.session_state.load_error = f"Transcript fetch failed: {type(e).__name__}: {e}"
        return None

    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant answering questions about a YouTube video.

Answer the question ONLY using the provided transcript context.
Do not use outside knowledge or make up information.
If the answer cannot be found in the context, say "I don't know."

Transcript Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    return parallel_chain | prompt | llm | StrOutputParser()


if go:
    st.session_state.current_video = video_id_input
    st.session_state.messages = []

if "current_video" in st.session_state:
    active_video_id = st.session_state.current_video
    chain = build_chain(active_video_id)

    if chain is None:
        st.error(st.session_state.get("load_error", "Failed to load this video."))
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for role, text in st.session_state.messages:
            st.chat_message(role).write(text)

        question = st.chat_input("Ask about the video")
        if question:
            st.session_state.messages.append(("user", question))
            st.chat_message("user").write(question)

            with st.spinner("Thinking..."):
                answer = chain.invoke(question)

            st.session_state.messages.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
else:
    st.info("Enter a video ID and click 'Load video' to start.")