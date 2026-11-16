import glob
import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Allow oauthlib to accept scope changes
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from google.auth.transport.requests import Request, AuthorizedSession
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/generative-language.peruserquota",
]

TOKEN_FILE = "token.json"


def find_client_secret_file():
    candidates = glob.glob("client_secret*.json")
    if candidates:
        return candidates[0]
    for f in os.listdir("."):
        if f.endswith(".json") and "client_secret" in f.lower():
            return f
    return None


def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        print(f"[INFO] Found existing {TOKEN_FILE}, loading credentials...")
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE)
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    print("[INFO] Token is expired, refreshing...")
                    creds.refresh(Request())
                    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                        f.write(creds.to_json())
        except Exception as e:
            print(f"[WARN] Failed to load {TOKEN_FILE}: {e}. Initiating fresh flow.")
            creds = None

    if not creds or not creds.valid:
        secret_file = find_client_secret_file()
        if not secret_file:
            print("[ERROR] No client_secret*.json file found in root directory!")
            sys.exit(1)

        print(f"[INFO] Starting OAuth 2.0 flow using {secret_file}...")
        print("[IMPORTANT] On Google consent screen, make sure to CHECK ALL BOXES for Drive, Docs, etc.!")
        flow = InstalledAppFlow.from_client_secrets_file(secret_file, SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent")

        with open(TOKEN_FILE, "w", encoding="utf-8") as token_out:
            token_out.write(creds.to_json())
        print(f"[SUCCESS] Credentials saved to {TOKEN_FILE}")

    print(f"[INFO] Granted scopes: {creds.scopes}")
    return creds


def test_user_info(creds):
    print("\n--- 1. Testing User Info (OAuth2) ---")
    try:
        oauth2_service = build("oauth2", "v2", credentials=creds)
        user_info = oauth2_service.userinfo().get().execute()
        email = user_info.get("email")
        name = user_info.get("name")
        print(f"[SUCCESS] Authenticated as: {name} ({email})")
        return True, email
    except Exception as e:
        print(f"[ERROR] Failed to fetch user info: {e}")
        return False, None


def test_drive_api(creds):
    print("\n--- 2. Testing Google Drive API ---")
    try:
        drive_service = build("drive", "v3", credentials=creds)
        results = (
            drive_service.files()
            .list(
                pageSize=3,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            )
            .execute()
        )
        files = results.get("files", [])
        print(f"[SUCCESS] Drive API connected. Found {len(files)} recent files:")
        for idx, f in enumerate(files, 1):
            print(f"   {idx}. {f.get('name')} (ID: {f.get('id')}, Type: {f.get('mimeType')})")
        return True, drive_service
    except Exception as e:
        print(f"[ERROR] Drive API error: {e}")
        return False, None


def test_docs_api(creds, drive_service=None):
    print("\n--- 3. Testing Google Docs API ---")
    doc_id = None
    try:
        docs_service = build("docs", "v1", credentials=creds)
        doc = docs_service.documents().create(body={"title": "Проверка CZECH_HARVESTER"}).execute()
        doc_id = doc.get("documentId")
        title = doc.get("title")
        print(f"[SUCCESS] Created Google Doc '{title}' with ID: {doc_id}")

        # Delete test document to clean up
        if doc_id:
            try:
                if not drive_service:
                    drive_service = build("drive", "v3", credentials=creds)
                drive_service.files().delete(fileId=doc_id).execute()
                print(f"[SUCCESS] Cleaned up: test document {doc_id} was successfully removed from Drive.")
            except Exception as del_err:
                print(f"[WARN] Could not automatically delete test doc: {del_err}")

        return True
    except Exception as e:
        print(f"[ERROR] Docs API error: {e}")
        return False


def test_gemini_api(creds):
    print("\n--- 4. Testing Gemini API ---")
    
    # 1. Test via OAuth token
    print("[INFO] Testing Gemini API via OAuth token...")
    authed_session = AuthorizedSession(creds)
    oauth_ok = False
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Ahoj! Napiš jednu krátkou větu v češtině o tom, že Google API funguje."}
                ]
            }
        ]
    }
    try:
        resp = authed_session.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            print(f"[SUCCESS] Gemini API responded via OAuth token:")
            print(f"   \"{text}\"")
            oauth_ok = True
        else:
            resp_json = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            reason = resp_json.get("error", {}).get("status", "")
            msg = resp_json.get("error", {}).get("message", resp.text)
            print(f"[NOTE] OAuth call returned HTTP {resp.status_code} ({reason}): {msg}")
            print("   Для вызова Gemini через OAuth в Google Cloud Console требуется скоуп:")
            print("   'https://www.googleapis.com/auth/generative-language' (вместо .peruserquota).")
    except Exception as e:
        print(f"[NOTE] OAuth request exception: {e}")

    # 2. Test via GEMINI_API_KEY from .env
    print("\n[INFO] Testing Gemini API via GEMINI_API_KEY from .env...")
    api_key = None
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as env_f:
            for line in env_f:
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break

    api_key_ok = False
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Ahoj! Napiš jednu krátkou větu v češtině o tom, že Google API funguje.",
            )
            print(f"[SUCCESS] Gemini API responded via GEMINI_API_KEY (.env):")
            print(f"   \"{response.text.strip()}\"")
            api_key_ok = True
        except Exception as e:
            print(f"[ERROR] Gemini API via API Key error: {e}")
    else:
        print("[WARN] GEMINI_API_KEY not found in .env")

    return oauth_ok, api_key_ok


def main():
    print("=" * 60)
    print("CZECH_HARVESTER Google OAuth & API Diagnostic")
    print("=" * 60)

    creds = get_credentials()

    status = {}
    ok_user, email = test_user_info(creds)
    status["UserInfo (OAuth2)"] = "OK" if ok_user else "FAILED"

    ok_drive, drive_service = test_drive_api(creds)
    status["Google Drive API"] = "OK" if ok_drive else "FAILED"

    ok_docs = test_docs_api(creds, drive_service)
    status["Google Docs API"]  = "OK" if ok_docs else "FAILED"

    oauth_gemini, apikey_gemini = test_gemini_api(creds)
    if apikey_gemini:
        status["Gemini API (API Key)"] = "OK"
    else:
        status["Gemini API (API Key)"] = "FAILED"
    
    if oauth_gemini:
        status["Gemini API (OAuth)"] = "OK"
    else:
        status["Gemini API (OAuth)"] = "SCOPE_INSUFFICIENT (needs generative-language scope)"

    print("\n" + "=" * 60)
    print("Diagnostic Summary:")
    for service, res in status.items():
        print(f"  - {service:25}: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
