from fastapi import APIRouter, BackgroundTasks
from app.services.evaluator import WorkflowEvaluator
from pydantic import BaseModel
from typing import List

router = APIRouter()
evaluator = WorkflowEvaluator()

class AdvancedEvaluationRequest(BaseModel):
    content_sample: str
    target_models: List[str]
    task_type: str = "Standard_Processing"
    enable_cache: bool = True
    strict_privacy: bool = False

@router.post("/evaluate")
async def evaluate_content(payload: AdvancedEvaluationRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        evaluator.process_matrix, 
        payload.content_sample, 
        payload.target_models,
        payload.task_type,
        payload.enable_cache,
        payload.strict_privacy
    )
    return {
        "status": "processing",
        "message": "Advanced VeloceAI matrix processing initiated with optimization profiling.",
        "configurations_tracked": {
            "cache_optimization": payload.enable_cache,
            "strict_privacy_gate": payload.strict_privacy,
            "target_profile": payload.task_type
        }
    }
