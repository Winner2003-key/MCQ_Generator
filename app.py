import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Question Generator",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False
if 'mcq_questions' not in st.session_state:
    st.session_state.mcq_questions = []
if 'short_questions' not in st.session_state:
    st.session_state.short_questions = []
if 'show_answers' not in st.session_state:
    st.session_state.show_answers = {}

def initialize_models():
    """Initialize the embedding model and LLM"""
    try:
        embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("EMBEDDING_API_BASE"),
            openai_api_key=os.getenv("EMBEDDING_API_KEY"),
            deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
            chunk_size=10,
        )
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=800,
            model_name="gpt-4o"
        )
        
        return embedding_model, llm
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

def process_document(file, file_type):
    """Process uploaded document and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load document based on file type
        if file_type == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_type == "txt":
            loader = TextLoader(tmp_file_path)
        else:
            st.error("Unsupported file type")
            return None
        
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return split_docs
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

def generate_questions(split_docs, embedding_model, llm, question_type):
    """Generate questions using the chain"""
    try:
        # Create vector store
        vector_db = FAISS.from_documents(split_docs, embedding_model)
        retriever = vector_db.as_retriever()
        
        # Define prompts based on question type
        if question_type == "mcq":
            prompt  = """
            Generate 5 MCQ questions in contextx with 4 options each (A, B, C, D) and indicate the correct answer.
            Return the response in valid JSON format like this:
            {
            "questions": [
                {
                "question": "What is...?",
                "options": {
                    "A": "Option A",
                    "B": "Option B", 
                    "C": "Option C",
                    "D": "Option D"
                },
                "correct_answer": "A",
                "explanation": "Explanation here"
                }
            ]
            }
            """
        else:
            prompt = """
                Generate 5 Short answer questions in context with detailed answers.
                Return the response in valid JSON format
                like this:
                {
                "questions": [
                    {
                    "question": "What is...?",
                    "answer": "Detailed answer here",
                    "key_points": ["Point 1", "Point 2"]
                    }
                ]
                }
                """
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever= retriever
        )
        
        # Generate questions
        #query = f"Generate {question_type} questions from the document"
        response = qa_chain.invoke(prompt)
        
        return response["result"]
        
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

def parse_questions_response(response):
    """Parse the JSON response from the LLM"""
    try:
        # Clean the response
        response = response.strip()
        
        # Remove markdown formatting if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1]
        
        # Find JSON content
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            questions_data = json.loads(json_str)
            return questions_data.get("questions", [])
        else:
            st.error("No valid JSON found in response")
            return []
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        st.error(f"Response received: {response[:500]}...")
        return []
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return []

def display_mcq_questions(questions):
    """Display MCQ questions"""
    for i, q in enumerate(questions):
        st.markdown(f"""
        <div class="question-box">
            <h4>Question {i+1}</h4>
            <p><strong>{q.get('question', 'No question available')}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display options
        options = q.get('options', {})
        for option_key, option_text in options.items():
            st.markdown(f"""
            <div class="option-text">
                <strong>{option_key})</strong> {option_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Show answer button
        answer_key = f"mcq_answer_{i}"
        if st.button(f"Show Answer", key=f"show_mcq_{i}"):
            st.session_state.show_answers[answer_key] = not st.session_state.show_answers.get(answer_key, False)
        
        # Display answer if toggled
        if st.session_state.show_answers.get(answer_key, False):
            correct_answer = q.get('correct_answer', 'N/A')
            explanation = q.get('explanation', 'No explanation provided')
            st.markdown(f"""
            <div class="answer-box">
                <p><strong>Correct Answer: {correct_answer}</strong></p>
                <p><strong>Explanation:</strong> {explanation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

def display_short_questions(questions):
    """Display short answer questions"""
    for i, q in enumerate(questions):
        st.markdown(f"""
        <div class="question-box">
            <h4>Question {i+1}</h4>
            <p><strong>{q.get('question', 'No question available')}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show answer button
        answer_key = f"short_answer_{i}"
        if st.button(f"Show Answer", key=f"show_short_{i}"):
            st.session_state.show_answers[answer_key] = not st.session_state.show_answers.get(answer_key, False)
        
        # Display answer if toggled
        if st.session_state.show_answers.get(answer_key, False):
            answer = q.get('answer', 'No answer provided')
            key_points = q.get('key_points', [])
            
            st.markdown(f"""
            <div class="answer-box">
                <p><strong>Answer:</strong> {answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if key_points:
                st.markdown("**Key Points:**")
                for point in key_points:
                    st.markdown(f"• {point}")
        
        st.markdown("---")

def main():
    # Simple header
    st.title("📝 Question Generator")
    st.markdown("Upload a document and generate MCQ or Short Answer questions")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt'],
            help="Upload a PDF or TXT file"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Generate questions button
            if st.button("Generate Questions", type="primary"):
                with st.spinner("Processing document and generating questions..."):
                    # Initialize models
                    embedding_model, llm = initialize_models()
                    
                    if embedding_model and llm:
                        # Process document
                        split_docs = process_document(uploaded_file, file_type)
                        
                        if split_docs:
                            st.info(f"Document processed: {len(split_docs)} chunks created")
                            
                            # Generate MCQ questions
                            with st.spinner("Generating MCQ questions..."):
                                mcq_response = generate_questions(split_docs, embedding_model, llm, "mcq")
                                if mcq_response:
                                    st.session_state.mcq_questions = parse_questions_response(mcq_response)
                            
                            # Generate short questions
                            with st.spinner("Generating short answer questions..."):
                                short_response = generate_questions(split_docs, embedding_model, llm, "short")
                                if short_response:
                                    st.session_state.short_questions = parse_questions_response(short_response)
                            
                            st.session_state.questions_generated = True
                            st.success("Questions generated successfully!")
                            st.rerun()
    
    # Main content area
    if not st.session_state.questions_generated:
        st.info("👈 Upload a document from the sidebar to start generating questions")
        st.markdown("**Supported formats:** PDF, TXT")
    else:
        # Tabs for different question types
        tab1, tab2 = st.tabs(["Multiple Choice Questions", "Short Answer Questions"])
        
        with tab1:
            st.subheader("Multiple Choice Questions")
            if st.session_state.mcq_questions:
                display_mcq_questions(st.session_state.mcq_questions)
            else:
                st.warning("No MCQ questions were generated. Try uploading the document again.")
        
        with tab2:
            st.subheader("Short Answer Questions")
            if st.session_state.short_questions:
                display_short_questions(st.session_state.short_questions)
            else:
                st.warning("No short answer questions were generated. Try uploading the document again.")
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset and Upload New Document"):
            st.session_state.questions_generated = False
            st.session_state.mcq_questions = []
            st.session_state.short_questions = []
            st.session_state.show_answers = {}
            st.rerun()

if __name__ == "__main__":
    main()
