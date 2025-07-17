import uvicorn
from src.api.api import create_app

app = create_app()

def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
