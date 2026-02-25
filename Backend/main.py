from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import httpx
import os

# -------------------------
# ENV
# -------------------------
load_dotenv()

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
RETRIEVAL_API_URL = os.getenv("RETRIEVAL_API_URL")

if not EMBEDDING_API_URL or not RETRIEVAL_API_URL:
    raise RuntimeError("Required environment variables are missing")

# -------------------------
# APP
# -------------------------
app = FastAPI(title="Knowledge Query Backend API")

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# TEMP STORE
# -------------------------
RESULT_STORE = {}
REQUEST_STATUS = {}

# -------------------------
# MODELS
# -------------------------
class ChatRequest(BaseModel):
    text: str
    top_k: int = 3

class ChatResponse(BaseModel):
    request_id: str
    status: str
    final_answer: str
    contexts_used: list[str]
    model: str
    timestamp: str

# -------------------------
# DATABASE / KB QUERY
# -------------------------
async def query_knowledge_base(payload: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                RETRIEVAL_API_URL,
                json=payload
            )
            response.raise_for_status()
            return response.json()

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Knowledge base query timed out")

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Knowledge base service unavailable")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge base error: {str(e)}")

# -------------------------
# MAIN QUERY ENDPOINT
# -------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    request_id = str(uuid.uuid4())
    REQUEST_STATUS[request_id] = "processing"

    # 1️⃣ Call embedding service
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            embed_response = await client.post(
                EMBEDDING_API_URL,
                json={"text": req.text}
            )
            embed_response.raise_for_status()
            embedding = embed_response.json().get("embedding")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding service error: {str(e)}")

    if not embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

    # 2️⃣ Build retrieval payload
    payload = {
        "request_id": request_id,
        "embedding": embedding,
        "top_k": req.top_k,
        "query": req.text   # 🔥 CRITICAL FIX — Send user question to RAG
    }

    # 3️⃣ Call retrieval service
    result = await query_knowledge_base(payload)

    # 4️⃣ Store result
    RESULT_STORE[request_id] = {
        "final_answer": result.get("final_answer", ""),
        "contexts": result.get("contexts_used", []),
        "model": result.get("model", ""),
        "timestamp": result.get("timestamp", "")
    }

    REQUEST_STATUS[request_id] = "completed"

    return ChatResponse(
        request_id=request_id,
        status="completed",
        final_answer=result.get("final_answer", ""),
        contexts_used=result.get("contexts_used", []),
        model=result.get("model", ""),
        timestamp=result.get("timestamp", "")
    )

# -------------------------
# STATUS CHECK
# -------------------------
@app.get("/status/{request_id}")
async def check_status(request_id: str):
    status = REQUEST_STATUS.get(request_id)

    if not status:
        raise HTTPException(status_code=404, detail="Request ID not found")

    response = {
        "request_id": request_id,
        "status": status
    }

    if status == "completed":
        response["result"] = RESULT_STORE.get(request_id)

    return response

# -------------------------
# HEALTH
# -------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_requests": len(
            [v for v in REQUEST_STATUS.values() if v == "processing"]
        ),
        "completed_requests": len(RESULT_STORE)
    }

# -------------------------
# ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
