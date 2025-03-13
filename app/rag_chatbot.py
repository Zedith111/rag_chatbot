import os
import uuid
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from unstructured.partition.pdf import partition_pdf

class RagChatbot:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        
        self.llm_model = ChatOllama(model="llama3.2", base_url=base_url)
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        self.image_model = ChatOllama(model="llama3.2-vision", base_url=base_url)
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

    def extractData(self):
        chunks = partition_pdf(
            filename=self.filepath,
            infer_table_structure=True,
            strategy="hi_res",

            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,

            chunking_strategy="by_title",
            max_characters=10000,
        )

        tables = []
        texts = []
        images = []

        for chunk in chunks:
            if "unstructured.documents.elements.Table" in str(type(chunk)):
                tables.append(chunk)
            
            if "unstructured.documents.elements.CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images.append(el.metadata.image_base64)
        
        self.tables = tables
        self.texts = texts
        self.images = images

    def summarizeData(self):
        self.summarizeText()
        self.summarizeImage()

    def summarizeText(self):
        prompt_template = """
        You are an assistant tasked to summarize table and text.
        Give a concise summary of the table or text.

        Respond only with summary, do not add additional comment or message, just give the summary

        Table or text = {chunk}        
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        summarize_chain = (
            {"chunk": lambda x:x}
            | prompt
            | self.llm_model
            | StrOutputParser()
        )

        self.text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 3})
        tables_html = [table.metadata.text_as_html for table in self.tables]
        self.table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    def summarizeImage(self):
        prompt_template = """
        Describe the image in detail
        """

        prompt = ChatPromptTemplate.from_template(
            "{prompt}\n\nImage: {image}"
        )

        image_summarize_chain = prompt | self.image_model | StrOutputParser()

        images = [
            {"prompt": prompt_template, "image": f"data:image/jpeg;base64,{image}"}
            for image in self.images
        ]

        self.image_summaries = image_summarize_chain.batch(images)
    
    def createVectorDb(self, file):
        self.storeFile(file)
        self.extractData()
        self.summarizeData()

        self.vector_db = Chroma(
            collection_name="rag",
            embedding_function=self.embedding_model,
            persist_directory=self.vector_db_dir
        )

        store = InMemoryStore()
        retriever = MultiVectorRetriever(
            vectorstore=self.vector_db,
            docstore=store,
            id_key="doc_id"
        )

        text_ids = [str(uuid.uuid4()) for _ in self.texts]
        summary_texts = [
            Document(page_content=summary, metadata={"doc_id": text_ids[i]})
            for i, summary in enumerate(self.text_summaries)
        ]
        if summary_texts:
            retriever.vectorstore.add_documents(summary_texts)
            retriever.docstore.mset(list(zip(text_ids,self.texts)))

        table_ids = [str(uuid.uuid4()) for _ in self.tables]
        summary_tables = [
            Document(page_content=summary, metadata={"doc_id": table_ids[i]})
            for i, summary in enumerate(self.table_summaries)
        ]
        if summary_tables:
            retriever.vectorstore.add_documents(summary_tables)
            retriever.docstore.mset(list(zip(table_ids,self.tables)))

        image_ids = [str(uuid.uuid4()) for _ in self.images]
        summary_images= [
            Document(page_content=summary, metadata={"doc_id": image_ids[i]})
            for i, summary in enumerate(self.image_summaries)
        ]
        if summary_images:
            retriever.vectorstore.add_documents(summary_images)
            retriever.docstore.mset(list(zip(text_ids,self.images)))

        self.retriever = retriever

    def chat(self, query):         
        prompt_template = """
            Answer the question based only following context, which can include text, tables and images
            If you don't know the answer,just say you don't know.
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
        


        

        

