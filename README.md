### EduGauge
An AI-powered app to predict learner skill level using NLP


### About the Project
Learnalyze brings personalization into education by analyzing user-submitted free-text answers to AI/ML/DL/GenAI questions. Instead of relying on MCQs, it uses semantic understanding to measure actual conceptual clarity.


### Key Features
Open-ended input for deeper knowledge assessment.

NLP-powered sentence embeddings using sentence-transformers.

Cosine similarity to measure understanding.

Learner classification (Beginner / Intermediate / Advanced).

Web interface built with Streamlit and hosted online.


### How It Works
User answers 10 AI-related questions in free text.

Each answer is embedded using a BERT-based model (MiniLM-L6-v2).

Compared against expert answers using cosine similarity.

Average score is calculated.

Based on score thresholds, learner level is assigned.


### Tech Stack                
Language   -    Python

Frontend	 -   Streamlit

AI Model	 -     sentence-transformers (MiniLM)   

Similarity  -   scikit-learn (cosine similarity) 

Hosting	    -  Streamlit Cloud + GitHub         



###  Installation
git clone https://github.com/rlokprasaath/EduGauge.git

pip install streamlit

streamlit run app.py


### Project Structure
├── app.py              

├── requirements.txt    

└── README.md           

