#!/usr/bin/env python3
"""Manual Google Calendar authentication for headless servers"""

import os
import sys
import pickle
from google_auth_oauthlib.flow import Flow

SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_PATH = os.path.expanduser("~/.jarvis_calendar_credentials.json")
TOKEN_PATH = os.path.expanduser("~/.jarvis_calendar_token.pickle")
REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'

def get_auth_url():
    """Get the authorization URL"""
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"ERROR: Credentials not found at {CREDENTIALS_PATH}")
        return None

    flow = Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    auth_url, _ = flow.authorization_url(prompt='consent')
    print("\n" + "="*60)
    print("GOOGLE CALENDAR AUTHORIZATION")
    print("="*60)
    print("\n1. Open this URL in your browser:\n")
    print(auth_url)
    print("\n2. Sign in and authorize the app")
    print("3. Copy the authorization code")
    print("4. Run: python3 auth_calendar.py YOUR_CODE_HERE")
    print("="*60 + "\n")
    return auth_url

def complete_auth(code):
    """Complete authentication with the authorization code"""
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    flow.fetch_token(code=code)
    creds = flow.credentials

    with open(TOKEN_PATH, 'wb') as token:
        pickle.dump(creds, token)

    print(f"\nSuccess! Token saved to {TOKEN_PATH}")
    print("You can now use Google Calendar with JARVIS.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Complete auth with provided code
        complete_auth(sys.argv[1])
    else:
        # Just show the auth URL
        get_auth_url()
