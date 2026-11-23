"""Generate private local demo credentials; never include them in Git."""
from pathlib import Path
from secrets import token_urlsafe

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".env"


def main():
    if DEST.exists():
        print(".env already exists; no changes made.")
        return
    contents = (
        f"CHURNGUARD_API_KEY={token_urlsafe(32)}\n"
        f"CHURNGUARD_UI_PASSWORD={token_urlsafe(32)}\n"
        "CHURNGUARD_API_URL=http://127.0.0.1:8000\n"
    )
    DEST.write_text(contents, encoding="utf-8")
    try:
        DEST.chmod(0o600)
    except OSError:
        pass
    print("Created .env with two generated credentials.")
    print("Open .env locally to view the UI password. Never commit it.")


if __name__ == "__main__":
    main()
