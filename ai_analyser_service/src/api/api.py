from fastapi import FastAPI
from src.models.models import AdviceRequest, AdviceResponse
from src.services.ai_service import AIAnalyzerService
from src.annotations.annotations import ai_service_exception_handler, health_check_exception_handler


ai_service = AIAnalyzerService()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Log Analyzer",
        description="AI-powered log analysis and troubleshooting advice service",
        version="1.0.0"
    )
    
    @app.post("/generate-advice", response_model=AdviceResponse)
    @ai_service_exception_handler
    async def generate_advice(request: AdviceRequest):
        advice = ai_service.generate_advice(request.events)
        return AdviceResponse(advice=advice)
    
    @app.get("/health")
    @health_check_exception_handler
    async def health_check():
        return {"status": "healthy", "service": "ai-log-analyzer"}
    
    return app
