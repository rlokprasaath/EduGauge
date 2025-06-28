import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------
# App Title
# ---------------------
st.set_page_config(page_title="AI Learner Type Predictor", layout="centered")
st.title("🤖 EduGauge: Measure. Learn. Improve.")
st.markdown("Answer the following questions to find out your learner level:")

# ---------------------
# Load Embedding Model (BERT-based)
# ---------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------------
# Questions & Reference Answers
# ---------------------
questions = [
    "1. What is Artificial Intelligence (AI), and where is it commonly used today?",
    "2. How is Machine Learning different from traditional programming?",
    "3. What is supervised learning, and can you give an example of its application?",
    "4. Explain what a neural network is in simple terms.",
    "5. What is the difference between deep learning and machine learning?",
    "6. What role do datasets play in training a machine learning model?",
    "7. What is overfitting in machine learning, and how can it be prevented?",
    "8. How do Generative AI models like ChatGPT or DALL·E work at a high level?",
    "9. What are large language models (LLMs), and what makes them powerful for GenAI tasks?",
    "10. Name two ethical concerns related to AI and explain why they matter."
]

reference_answers = [
    "Artificial Intelligence is the simulation of human intelligence by machines. It's used in applications like virtual assistants, recommendation systems, self-driving cars, facial recognition, and fraud detection.",
    "In traditional programming, humans write rules and logic. In machine learning, the system learns patterns and rules automatically from data to make predictions or decisions.",
    "Supervised learning is a type of machine learning where models are trained on labeled data. An example is spam detection in emails, where each email is labeled as spam or not.",
    "A neural network is a computational model inspired by the human brain, made up of interconnected nodes (neurons) that process input data and learn to recognize patterns for tasks like classification and prediction.",
    "Deep learning is a subset of machine learning that uses multi-layered neural networks to learn complex patterns from large datasets, while machine learning includes a broader range of algorithms like decision trees or SVMs.",
    "Datasets provide the input examples from which the model learns. The quality and size of the dataset directly affect the model’s performance and generalization ability.",
    "Overfitting happens when a model learns noise in the training data and fails to generalize to new data. It can be prevented using techniques like regularization, cross-validation, or using more training data.",
    "Generative AI models learn from vast datasets to understand structure and context, then generate new content by predicting sequences — such as text in ChatGPT or images in DALL·E — using deep neural networks like transformers.",
    "LLMs are deep learning models trained on massive text data to understand and generate human-like language. Their power comes from their scale and ability to capture nuanced linguistic patterns and context.",
    "Two major concerns are bias in AI decisions and lack of transparency. Biased models can reinforce discrimination, and opaque decision-making limits accountability in critical applications like healthcare or law."
]

user_answers = []

# ---------------------
# Ask Questions
# ---------------------
for i, q in enumerate(questions):
    ans = st.text_area(q, key=f"user_q{i}")
    user_answers.append(ans)

# ---------------------
# Submit & Process
# ---------------------
if st.button("Submit Answers"):
    if all(a.strip() != "" for a in user_answers):
        # Embed both sets of answers
        ref_embeddings = model.encode(reference_answers)
        user_embeddings = model.encode(user_answers)

        # Compare similarity per question
        similarities = []
        for u, r in zip(user_embeddings, ref_embeddings):
            sim = cosine_similarity([u], [r])[0][0]
            similarities.append(sim)

        avg_similarity = np.mean(similarities)

        st.markdown("---")
        st.subheader("📊 Results")
        for i, score in enumerate(similarities):
            st.write(f"**Q{i+1} Similarity:** {score:.2f}")

        st.write(f"**Average Similarity:** {avg_similarity:.2f}")

        # Classify learner type
        if avg_similarity < 0.4:
            level = "🟢 Beginner"
        elif avg_similarity < 0.7:
            level = "🟡 Intermediate"
        else:
            level = "🔵 Advanced"

        st.success(f"### Your predicted learner type is: {level}")
        
    else:
        st.warning("Please answer all questions before submitting.")
