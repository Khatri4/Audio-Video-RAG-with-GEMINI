# 🎥🎧 Video and Audio RAG

Welcome to **Video and Audio RAG**! This innovative application lets you interact with your video and audio files through advanced transcription and question-answering capabilities. Dive into your media content with ease and precision! 🧠🔍

## 🚀 Features

- **📂 Transcribe Media**: Upload and transcribe multiple video or audio files (but not both at the same time).
- **💬 Interactive Q&A**: Ask detailed questions about your transcribed content and receive precise answers.
- **🔍 Efficient Search**: Fast and accurate similarity search within your transcriptions using FAISS.

## 🛠️ Technologies Used

This project leverages a variety of powerful libraries and technologies:

- **🌐 [AssemblyAI](https://assemblyai.com/)**: Transcribes audio and video files to text.
- **🧠 [Google Generative AI](https://developers.google.com/ai)**: Provides embeddings and generative AI capabilities.
- **📚 [LangChain](https://www.langchain.com/)**: Facilitates chaining of LLMs for sophisticated question-answering.
- **🔎 [FAISS](https://faiss.ai/)**: Efficiently handles vector similarity search.
- **🔄 [Streamlit](https://streamlit.io/)**: Powers the interactive web interface.

## 🛠️ Getting Started

Follow these steps to set up and use **Video and Audio RAG**:

### 1. Clone the Repository

```bash
git clone https://github.com/Khatri4/video-audio-rag.git
cd video-audio-rag
```

### 2. Set Up Your Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Your API Keys

To enable transcription and embedding services, set up your API keys. Create a `.env` file in the root directory and add your keys:

```env
ASSEMBLY_API=your_assemblyai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Replace `your_assemblyai_api_key_here` and `your_google_api_key_here` with your actual API keys.

### 4. Run the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

Visit the URL provided by Streamlit to access the application in your browser.

## 📂 How to Use

1. **🔧 Select File Type**: Choose between "Video" or "Audio" for file uploads.
2. **📤 Upload Files**: Use the sidebar file uploader to select and upload multiple files of the chosen type.
3. **❓ Ask Questions**: Enter your questions about the content and get detailed responses based on your uploaded media.

## 📝 Troubleshooting

- **🔑 API Key Issues**: Ensure your API keys are correct and active.
- **⚠️ File Type Errors**: Verify that all uploaded files match the selected type (either all video or all audio).

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
