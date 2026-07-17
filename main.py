from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from src.app import run_app
    run_app()