from fastapi import APIRouter, BackgroundTasks
from app.services.evaluator import WorkflowEvaluator
from pydantic import BaseModel
from typing import List

router = APIRouter()
evaluator = WorkflowEvaluator()

class EvaluationRequest(BaseModel):
    content_sample: str
    target_models: List[str]
    runpod_endpoint_id: str = None

@router.post("/evaluate")
async def evaluate_content(payload: EvaluationRequest, background_tasks: BackgroundTasks):
    # Runs the calculation matrix safely in background thread to preserve server event loops
    background_tasks.add_task(
        evaluator.process_matrix, 
        payload.content_sample, 
        payload.target_models
    )
    return {
        "status": "processing",
        "message": "VeloceAI matrix processing tracking initiated in background.",
        "models_evaluated": payload.target_models
    }
