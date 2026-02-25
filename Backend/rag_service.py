from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from datetime import datetime
from groq import Groq

# ===============================
# ENV
# ===============================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "playbook")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")

# ===============================
# APP + DB
# ===============================
app = FastAPI(title="RAG Service with Groq")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

client = Groq(api_key=GROQ_API_KEY)

# ===============================
# REQUEST MODEL
# ===============================
class ChatRequest(BaseModel):
    request_id: Optional[str] = None
    embedding: List[float]
    top_k: Optional[int] = 5
    query: Optional[str] = None

# ===============================
# RESPONSE MODEL
# ===============================
class ChatResponse(BaseModel):
    request_id: Optional[str]
    status: str
    final_answer: str
    contexts_used: List[str]
    relevance_scores: List[float]
    model: str
    timestamp: str

# ===============================
# LLM CALL
# ===============================
def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Security Knowledge & Playbook Assistant. "
                    "Follow these rules:\n"
                    "- If the user's query is a greeting, respond politely.\n"
                    "- If the question is a general cybersecurity definition (e.g., phishing, malware), give a simple explanation.\n"
                    "- If context from playbooks is provided, answer strictly using that context.\n"
                    "- Provide structured answers when applicable: Title, Pre-checks, Steps, Escalation, Post-Incident.\n"
                    "- If context is missing but the user asks something SOC-related, provide a helpful general response."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=350
    )
    return response.choices[0].message.content.strip()

# ===============================
# BUILT-IN FALLBACK KNOWLEDGE
# ===============================
BASIC_FAQ = {
    "what is phishing": "Phishing is a social engineering attack where attackers trick users into revealing sensitive information by pretending to be trusted entities.",
    "what is malware": "Malware is malicious software designed to disrupt, damage, or gain unauthorized access to a computer system.",
    "what is ransomware": "Ransomware is malware that encrypts a victim's files and demands payment for the decryption key.",
    "what is firewall": "A firewall is a security device/software that monitors and filters incoming and outgoing network traffic to protect systems.",
    "what is brute force": "A brute-force attack attempts to guess passwords or keys by trying many combinations until it succeeds.",
    "what is ddos": "A DDoS attack overloads a target using a flood of traffic from multiple systems, disrupting availability."
}

GREETINGS = ["hi", "hello", "hey", "hola", "yo", "good morning", "good evening"]

# ===============================
# CHAT ENDPOINT
# ===============================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    if not req.embedding:
        raise HTTPException(status_code=400, detail="Embedding missing")

    # Lower threshold for Pinecone match
    RELEVANCE_THRESHOLD = 0.35
    top_k = min(req.top_k or 5, 10)

    # Normalize query text
    user_query = (req.query or "").lower().strip()

    # 1️⃣ GREETING FALLBACK
    if user_query in GREETINGS:
        return ChatResponse(
            request_id=req.request_id,
            status="completed",
            final_answer="Hello! I'm your SOC Assistant. How can I help you today?",
            contexts_used=[],
            relevance_scores=[],
            model="fallback-greeting",
            timestamp=datetime.utcnow().isoformat()
        )

    # 2️⃣ FAQ FALLBACK
    for key in BASIC_FAQ:
        if key in user_query:
            return ChatResponse(
                request_id=req.request_id,
                status="completed",
                final_answer=BASIC_FAQ[key],
                contexts_used=[],
                relevance_scores=[],
                model="fallback-faq",
                timestamp=datetime.utcnow().isoformat()
            )

    # 3️⃣ PINECONE SEARCH
    results = index.query(
        vector=req.embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = results.get("matches", [])
    contexts = []
    scores = []

    for m in matches:
        score = m.get("score", 0)
        if score < RELEVANCE_THRESHOLD:
            continue

        metadata = m.get("metadata", {})
        content = metadata.get("content") or metadata.get("text")

        if content:
            contexts.append(content)
            scores.append(score)

    # 4️⃣ IF NO RAG CONTEXT → SOC GENERAL FALLBACK
    if not contexts:
        return ChatResponse(
            request_id=req.request_id,
            status="completed",
            final_answer=(
                "I couldn't find related playbook context, but here's a general SOC explanation:\n\n"
                "• Ensure proper log sources are enabled.\n"
                "• Investigate alerts using SIEM.\n"
                "• Validate whether behavior is malicious.\n"
                "• Escalate if suspicious.\n\n"
                "You can also ask me playbook questions like:\n"
                "• How to handle phishing?\n"
                "• Malware investigation steps?"
            ),
            contexts_used=[],
            relevance_scores=[],
            model="fallback-general",
            timestamp=datetime.utcnow().isoformat()
        )

    # 5️⃣ FORM PLAYBOOK CONTEXT PROMPT
    context_text = "\n\n".join([c[:700] for c in contexts[:2]])

    prompt = f"""
Context:
{context_text}

User Question:
{user_query}

Provide the best possible answer based on the context.
Use a structured SOC format when applicable.
"""

    # 6️⃣ CALL LLM
    final_answer = call_llm(prompt)

    return ChatResponse(
        request_id=req.request_id,
        status="completed",
        final_answer=final_answer,
        contexts_used=contexts[:2],
        relevance_scores=scores[:2],
        model="llama-3.1-8b-instant",
        timestamp=datetime.utcnow().isoformat()
    )

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "llama-3.1-8b-instant",
        "index": INDEX_NAME
    }
