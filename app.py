import streamlit as st
import fitz  # PyMuPDF
import easyocr
import tempfile
from PIL import Image
import re
import os
from sentence_transformers import SentenceTransformer, util
import openai

# ---- OpenAI Setup ----
openai.api_key = "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"

# ---- NLP Model for Similarity ----
model = SentenceTransformer('all-MiniLM-L6-v2')
ocr_reader = easyocr.Reader(['en'])

# ---- Streamlit UI ----
st.title("📚 Smart Question Scanner")
st.write("Upload old question papers (PDFs or Images) to detect repeated questions and topics.")

uploaded_files = st.file_uploader("Upload Files (PDF or Image)", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_files:
    all_questions = []

    for file in uploaded_files:
        file_type = file.type

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        text = ""

        if 'pdf' in file_type:
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
        elif 'image' in file_type:
            image = Image.open(tmp_path)
            text = ocr_reader.readtext(np.array(image), detail=0)
            text = ' '.join(text)

        os.unlink(tmp_path)  # cleanup

        # Split into questions
        raw_questions = re.split(r'\n|\d{1,2}\.\s+|\?\s+', text)
        questions = [q.strip() for q in raw_questions if len(q.strip()) > 15]
        all_questions.extend(questions)

    # Embeddings & Similarity
    st.write("🔍 Processing Questions...")
    embeddings = model.encode(all_questions, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    grouped = []
    used = set()

    for i, q in enumerate(all_questions):
        if i in used:
            continue
        group = [q]
        used.add(i)
        for j in range(i+1, len(all_questions)):
            if sim_matrix[i][j] > 0.85 and j not in used:
                group.append(all_questions[j])
                used.add(j)
        grouped.append(group)

    results = []

    # Tag Topics using GPT
    st.write("🧠 Tagging topics with AI...")
    for group in grouped:
        question = group[0]
        count = len(group)

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful assistant that tags questions by academic topic."},
                {"role": "user", "content": f"What is the subject/topic of this question? Respond in 1-2 words.\n\nQ: {question}"}
            ]
        )
        topic = response['choices'][0]['message']['content'].strip()
        results.append((question, count, topic))

    # Show Results
    st.success("✅ Done! Here are your most repeated questions:")

    results = sorted(results, key=lambda x: x[1], reverse=True)
    for i, (q, count, topic) in enumerate(results):
        st.markdown(f"**{i+1}. ({count} times)** — *{topic}*  \n{q}")

