import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import os
import uuid
import base64
import streamlit as st
from PIL import Image
from io import BytesIO

# --- Import your libraries ---
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# --- Define helper functions ---

def load_pdf_and_process(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

def extract_elements(chunks):
    texts = []
    tables = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    return texts, tables

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def generate_image_summaries(images_list):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image_summaries = []
    for image_b64 in images_list:
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.generate(**inputs)
            description = processor.decode(outputs[0], skip_special_tokens=True)
            image_summaries.append(description)
        except Exception as e:
            st.error(f"Error processing an image: {e}")
    return image_summaries

# --- Custom Document Store & Embeddings (from your code) ---


class CustomInMemoryDocStore(BaseStore[str, Document]):
    def __init__(self):
        self._store = {}
    def mset(self, key_value_pairs):
        for key, value in key_value_pairs:
            self._store[key] = value
    def mget(self, keys):
        return [self._store.get(key) for key in keys]
    def mdelete(self, keys):
        for key in keys:
            self._store.pop(key, None)
    def yield_keys(self, prefix=None):
        for key in self._store:
            if prefix is None or key.startswith(prefix):
                yield key

class OpenSourceEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, docs):
        processed_docs = [doc if (isinstance(doc, str) and doc.strip()) else "unknown" for doc in docs]
        return self.model.encode(processed_docs, convert_to_numpy=True).tolist()
    def embed_query(self, query: str):
        query = query if (isinstance(query, str) and query.strip()) else "unknown"
        return self.model.encode([query], convert_to_numpy=True).tolist()[0]

def index_documents(summaries, originals, doc_type, vectorstore, docstore, id_key):
    doc_ids = []
    documents = []
    for summary, original in zip(summaries, originals):
        if summary and summary.strip():
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            documents.append(Document(page_content=summary, metadata={id_key: doc_id, "type": doc_type}))
    if documents:
        vectorstore.add_documents(documents)
        for doc_id, original in zip(doc_ids, originals):
            content = original if (isinstance(original, str) and original.strip()) else "unknown"
            docstore.mset([(doc_id, Document(page_content=content, metadata={"type": doc_type}))])
    else:
        st.write(f"No valid {doc_type} documents to add.")

def build_vectorstore(text_summaries, texts, table_summaries, tables, image_summaries):
    embedding_function = OpenSourceEmbeddings()
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_function)
    docstore = CustomInMemoryDocStore()
    id_key = "doc_id"

    text_originals = [str(t) for t in texts]
    table_originals = [str(t) for t in tables]
    image_originals = [str(i) for i in image_summaries]

    index_documents(text_summaries, text_originals, "text", vectorstore, docstore, id_key)
    index_documents(table_summaries, table_originals, "table", vectorstore, docstore, id_key)
    index_documents(image_summaries, image_originals, "image", vectorstore, docstore, id_key)

    return vectorstore, docstore, id_key

def get_retriever(vectorstore, docstore, id_key):
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": 5}
    )
    return retriever

def run_retrieval_qa(query, retriever):
    llm = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return rag_pipeline(query)

# --- Streamlit App Interface ---

def main():
    st.title("Multi-Modal Retrieval QA System")
    st.write("Upload a PDF file and ask questions about its contents.")
    GROQ_API_KEY_SECRET="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"
    LANGCHAIN_API_KEY= "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"
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                chunks = load_pdf_and_process("temp.pdf")
                texts, tables = extract_elements(chunks)
                images_list = get_images_base64(chunks)
                
                # For demonstration, we use simple truncation for summaries.
                text_summaries = [str(t)[:200] for t in texts]
                table_summaries = [str(t)[:200] for t in tables]
                image_summaries = generate_image_summaries(images_list)
                
                vectorstore, docstore, id_key = build_vectorstore(
                    text_summaries, texts, table_summaries, tables, image_summaries
                )
                retriever = get_retriever(vectorstore, docstore, id_key)
                st.session_state['retriever'] = retriever
                st.success("PDF processed and indexed!")
                
        query = st.text_input("Enter your query:")
        if st.button("Run Query") and query:
            retriever = st.session_state.get('retriever')
            if retriever is None:
                st.error("Please process a PDF first.")
            else:
                result = run_retrieval_qa(query, retriever)
                st.write("**Answer:**", result['result'])
                st.write("**Source Documents:**")
                for doc in result['source_documents']:
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()
