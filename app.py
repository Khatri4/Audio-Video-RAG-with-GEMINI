import os
import streamlit as st
import assemblyai as aai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set your AssemblyAI API key
aai.settings.api_key = os.getenv("ASSEMBLY_API")

# Google Generative AI configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def transcribe_file(va_docs):
    transcripts = []
    transcriber = aai.Transcriber()
    for va in va_docs:
        transcript = transcriber.transcribe(va)
        transcripts.append(transcript.text)
    return '\n'.join(transcripts)


def get_text_chunks(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(transcript)
    return chunks


def get_vector_store(transcript_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(transcript_chunks, embedding=embeddings)
    vector_store.save_local("faissDB")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided
    context just say. "answer is not available in the context", don't provide the wrong answer\n\n
    Context: \n {context}?\n
    Question: \n {question}\n

    Answer
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faissDB", embeddings, allow_dangerous_deserialization=True)
    transcriptions = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": transcriptions, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Video and Audio RAG")
    st.header("Chat with your Video and Audio")

    # Selector to choose between video or audio
    file_type = st.selectbox("Select file type:", ["Video", "Audio"])

    user_question = st.text_input("Ask your question: ")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("File Upload")

        if file_type == "Video":
            va_docs = st.file_uploader("Upload video files", type=["mp4", "avi", "mov", "mkv"],
                                       accept_multiple_files=True)
        elif file_type == "Audio":
            va_docs = st.file_uploader("Upload audio files", type=["wav", "mp3", "ogg", "flac"],
                                       accept_multiple_files=True)

        if st.button("Submit"):
            if va_docs:
                # Check if all files are of the selected type
                if file_type == "Video" and all(f.name.endswith((".mp4", ".avi", ".mov", ".mkv")) for f in va_docs):
                    with st.spinner("Processing..."):
                        raw_text = transcribe_file(va_docs)
                        transcribe_chunk = get_text_chunks(raw_text)
                        get_vector_store(transcribe_chunk)
                        st.success("Done")
                elif file_type == "Audio" and all(f.name.endswith((".wav", ".mp3", ".ogg", ".flac")) for f in va_docs):
                    with st.spinner("Processing..."):
                        raw_text = transcribe_file(va_docs)
                        transcribe_chunk = get_text_chunks(raw_text)
                        get_vector_store(transcribe_chunk)
                        st.success("Done")
                else:
                    st.error("Please upload files of the selected type only.")
            else:
                st.error("Please upload files before submitting.")


if __name__ == "__main__":
    main()
