import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader

class RagChatbot:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        
        self.llm_model = ChatOllama(model="llama3.2", base_url=base_url)
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        self.file_storage_dir = os.path.join(current_dir, "data")
        self.vector_db_dir = os.path.join(current_dir, "db")
        self.vector_db = None
        self.retriever = None

        os.makedirs(self.file_storage_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

    def storeFile(self, file):
        filepath = os.path.join(self.file_storage_dir, file.name)

        with open(filepath, "wb") as f:
            f.write(file.getvalue())

        if not os.path.exists(filepath):
            raise Exception("Unable to save file")
        
        self.filepath = filepath

    def createVectorDb(self, file):
        self.storeFile(file)

        reader = PdfReader(self.filepath)

        content = ""
        for page in reader.pages:
            content += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

        chunks = text_splitter.split_text(content)

        self.vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=self.embedding_model,
            persist_directory=self.vector_db_dir
        )

    def createRetriever(self):
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                You are a AI asistant.Your task is to generate 3 different version of the given question to retrieve relevant
                documents from a vector database.

                By generating multiple perspectives on the user question, your goal is to help the user overcome some of the 
                limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}
            """
        )

        self.retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(), self.llm_model, prompt=query_prompt
        )

    def chat(self, query): 
        self.createRetriever()
        
        prompt_template = """
            Answer the question based only following context.
            If you don't k ow the answer,just say you don't know.
            Yor answer should be short and concise
            {context}
            Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | prompt 
            | self.llm_model 
            | StrOutputParser()
        )

        response = chain.invoke(query)

        return response
        


        

        

