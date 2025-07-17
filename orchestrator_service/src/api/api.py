from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from src.config.config import (
    PARSER_SERVICE_URL, 
    AI_ANALYZER_SERVICE_URL,
    PARSER_SERVICE_TIMEOUT,
    AI_ANALYZER_SERVICE_TIMEOUT,
    HEALTH_CHECK_TIMEOUT
)
from src.annotations.annotations import parser_service_exception_handler, ai_analyzer_service_exception_handler
import httpx
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@parser_service_exception_handler
async def parse_logs(log_content: bytes) -> List[Dict[str, Any]]:
    """Send log file to parser service and get parsed events."""
    async with httpx.AsyncClient() as client:
        files = {"file": ("log.txt", log_content, "text/plain")}
        response = await client.post(
            f"{PARSER_SERVICE_URL}/parse_log/",
            files=files,
            timeout=PARSER_SERVICE_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["events"]


@ai_analyzer_service_exception_handler
async def analyze_events(events: List[Dict[str, Any]]) -> str:
    """Send parsed events to AI analyzer service and get analysis."""
    async with httpx.AsyncClient() as client:
        payload = {"events": events}
        response = await client.post(
            f"{AI_ANALYZER_SERVICE_URL}/generate-advice",
            json=payload,
            timeout=AI_ANALYZER_SERVICE_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["advice"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Log Analysis Orchestrator",
        description="Orchestrates log parsing and AI analysis services",
        version="1.0.0"
    )

    @app.post("/analyze_log/")
    async def analyze_log(file: UploadFile = File(...)):
        """
        Complete log analysis pipeline:
        1. Parse log file using parser service
        2. Analyze parsed events using AI analyzer service
        3. Return analysis results
        """
        try:
            content = await file.read()
            logger.info("Parsing log file...")
            parsed_events = await parse_logs(content)
            
            if not parsed_events:
                return JSONResponse(content={
                    "events": [],
                    "analysis": "No error events found in the log file."
                })
            logger.info(f"Analyzing {len(parsed_events)} events...")
            analysis = await analyze_events(parsed_events)
            
            return JSONResponse(content={
                "analysis": analysis
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in analyze_log: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze log: {str(e)}"
            )

    @app.get("/health")
    async def health_check():
        """Health check endpoint that also verifies downstream services."""
        try:
            async with httpx.AsyncClient() as client:
                parser_response = await client.get(
                    f"{PARSER_SERVICE_URL}/health",
                    timeout=HEALTH_CHECK_TIMEOUT
                )
                parser_healthy = parser_response.status_code == 200
        
                ai_response = await client.get(
                    f"{AI_ANALYZER_SERVICE_URL}/health",
                    timeout=HEALTH_CHECK_TIMEOUT
                )
                ai_healthy = ai_response.status_code == 200
                
                return {
                    "status": "healthy" if parser_healthy and ai_healthy else "degraded",
                    "services": {
                        "parser": "healthy" if parser_healthy else "unhealthy",
                        "ai_analyzer": "healthy" if ai_healthy else "unhealthy"
                    }
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "Log Analysis Orchestrator",
            "version": "1.0.0",
            "description": "Orchestrates log parsing and AI analysis services",
            "endpoints": {
                "analyze_log": "POST /analyze_log/ - Upload log file for complete analysis",
                "health": "GET /health - Health check including downstream services"
            }
        }

    return app
