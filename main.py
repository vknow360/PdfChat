from dotenv import load_dotenv
from uvicorn import run

load_dotenv()

if __name__ == "__main__":
    run("src.app:app", host="127.0.0.1", port=8000, reload=True)