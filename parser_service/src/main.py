from models.log_event import LogLevel
from parser.parser_factory import get_parser
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File
import uvicorn

app = FastAPI()

@app.post("/parse_log/")
async def parse_log(file: UploadFile = File(...)):
    content = await file.read()
    raw_log = content.decode("utf-8")
    parser = get_parser("circleci")
    messages = parser.parse(raw_log)
    errors = [
        {
            "timestamp": m.timestamp,
            "message": m.message,
            "level": m.level.name,
            "source": m.source
        }
        for m in messages if m.level == LogLevel.ERROR
    ]
    return JSONResponse(content={"events": errors})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy", "service": "parser"})


def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
