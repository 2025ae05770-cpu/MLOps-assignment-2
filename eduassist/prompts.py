"""The three assignment templates and a fixed academic evaluation set."""

PROMPTS = {
    "v1": {
        "template": "Answer the following student question clearly.\nQuestion: {{question}}",
        "commit_message": "Basic baseline: clear student question answering without output constraints.",
    },
    "v2": {
        "template": (
            "You are an academic assistant.\n"
            "Answer the following student question in a clear, concise, and accurate manner.\n"
            "Use simple language suitable for undergraduate students.\n"
            "Question: {{question}}"
        ),
        "commit_message": "Structured guidance: prioritize concise, accurate undergraduate explanations.",
    },
    "v3": {
        "template": (
            "You are an academic assistant.\n"
            "Answer the student question and return ONLY valid JSON.\n"
            "Do not include markdown, explanations outside JSON, or code fences.\n"
            "Use this exact JSON schema:\n"
            '{\n  "answer": "string",\n  "confidence": "high | medium | low",\n'
            '  "key_points": ["string"]\n}\n'
            "Question: {{question}}"
        ),
        "commit_message": "JSON contract: answer, confidence enum, and string key points without markdown.",
    },
}

QUESTIONS = [
    {"question_id": "Q1", "question": "What is overfitting in machine learning, and how can it be reduced?",
     "reference": "Overfitting fits training noise and generalizes poorly. Use validation, regularization, simpler models, or more representative data."},
    {"question_id": "Q2", "question": "Explain the difference between supervised and unsupervised learning with examples.",
     "reference": "Supervised learning uses labeled examples, such as classification or regression. Unsupervised learning finds structure without labels, such as clustering."},
    {"question_id": "Q3", "question": "What is gradient descent, and what role does the learning rate play?",
     "reference": "Gradient descent updates parameters opposite the loss gradient. Too large a learning rate may overshoot or diverge; too small can converge slowly."},
    {"question_id": "Q4", "question": "Why are precision and recall important for evaluating a classifier?",
     "reference": "Precision is TP/(TP+FP); recall is TP/(TP+FN). They expose false-positive and false-negative tradeoffs, especially with imbalanced data."},
    {"question_id": "Q5", "question": "How does LLMOps differ from traditional MLOps?",
     "reference": "MLOps manages data, training, model deployment, and monitoring. LLMOps additionally versions prompts, retrieval, API models and guardrails, and measures tokens, cost, latency, and output quality."},
]