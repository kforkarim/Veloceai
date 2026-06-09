import time
import httpx
import litellm
from app.core.config import RUNPOD_API_KEY, LITELLM_MASTER_KEY

class WorkflowEvaluator:
    def __init__(self):
        self.runpod_bearer = RUNPOD_API_KEY
        litellm.api_key = LITELLM_MASTER_KEY

    async def trigger_runpod_serverless(self, endpoint_id: str, payload: dict) -> dict:
        """Kicks off a serverless GPU job on RunPod and monitors execution stats."""
        url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        headers = {
            "Authorization": f"Bearer {self.runpod_bearer}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            # Trigger job
            response = await client.post(url, json={"input": payload}, headers=headers, timeout=10.0)
            if response.status_code != 200:
                return {"status": "failed", "error": "Could not spin up RunPod worker"}
            
            job_data = response.json()
            job_id = job_data.get("id")
            
            # Poll for job metrics completion
            status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
            while True:
                status_check = await client.get(status_url, headers=headers)
                status_res = status_check.json()
                
                if status_res.get("status") == "COMPLETED":
                    execution_time = time.time() - start_time
                    output = status_res.get("output", {})
                    return {
                        "status": "completed",
                        "execution_time_seconds": round(execution_time, 2),
                        "tokens_processed": output.get("usage", {}).get("total_tokens", 0),
                        "raw_output": output.get("result", "")
                    }
                elif status_res.get("status") in ["FAILED", "CANCELLED"]:
                    return {"status": "failed", "error": "Job failed on GPU worker"}
                
                await asyncio.sleep(1.0)

    async def process_matrix(self, sample_content: str, model_list: list) -> dict:
        """Evaluates sample data across multi-model variations using unified LiteLLM tracking."""
        results = {}
        for model in model_list:
            start_ttft = time.time()
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": f"Analyze this text sample: {sample_content}"}]
                )
                duration = time.time() - start_ttft
                results[model] = {
                    "status": "success",
                    "latency_seconds": round(duration, 3),
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "response": response.choices[0].message.content
                }
            except Exception as e:
                results[model] = {"status": "error", "message": str(e)}
        return results
