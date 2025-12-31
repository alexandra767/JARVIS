#!/usr/bin/env python3
"""
JARVIS Calendar & Email Integration
- Google Calendar integration
- Email reading/summarization
- Reminders system
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import threading
import time

logger = logging.getLogger("JarvisCalendar")

# ============================================================================
# LOCAL REMINDERS (No external service needed)
# ============================================================================

@dataclass
class Reminder:
    """A reminder"""
    id: str
    message: str
    remind_at: datetime
    created_at: datetime
    completed: bool = False
    spoken: bool = False


class ReminderManager:
    """Manage local reminders"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.expanduser("~/.jarvis_reminders.json")
        self.reminders: Dict[str, Reminder] = {}
        self._counter = 0
        self.running = False
        self._thread = None

        # Callbacks
        self.on_reminder: Optional[callable] = None

        self._load()

    def _load(self):
        """Load reminders from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    r = Reminder(
                        id=item["id"],
                        message=item["message"],
                        remind_at=datetime.fromisoformat(item["remind_at"]),
                        created_at=datetime.fromisoformat(item["created_at"]),
                        completed=item.get("completed", False),
                        spoken=item.get("spoken", False)
                    )
                    self.reminders[r.id] = r
                    self._counter = max(self._counter, int(r.id.split("_")[1]) + 1)
        except Exception as e:
            logger.warning(f"Could not load reminders: {e}")

    def _save(self):
        """Save reminders to disk"""
        try:
            data = [
                {
                    "id": r.id,
                    "message": r.message,
                    "remind_at": r.remind_at.isoformat(),
                    "created_at": r.created_at.isoformat(),
                    "completed": r.completed,
                    "spoken": r.spoken
                }
                for r in self.reminders.values()
            ]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Could not save reminders: {e}")

    def add_reminder(self, message: str, remind_at: datetime = None,
                     delay_minutes: int = None) -> Reminder:
        """Add a new reminder"""
        self._counter += 1
        reminder_id = f"reminder_{self._counter}"

        if remind_at is None and delay_minutes:
            remind_at = datetime.now() + timedelta(minutes=delay_minutes)
        elif remind_at is None:
            remind_at = datetime.now() + timedelta(hours=1)  # Default 1 hour

        reminder = Reminder(
            id=reminder_id,
            message=message,
            remind_at=remind_at,
            created_at=datetime.now()
        )

        self.reminders[reminder_id] = reminder
        self._save()

        logger.info(f"Added reminder: {message} at {remind_at}")
        return reminder

    def remove_reminder(self, reminder_id: str) -> bool:
        """Remove a reminder"""
        if reminder_id in self.reminders:
            del self.reminders[reminder_id]
            self._save()
            return True
        return False

    def get_pending(self) -> List[Reminder]:
        """Get all pending reminders"""
        now = datetime.now()
        return [
            r for r in self.reminders.values()
            if not r.completed and r.remind_at <= now
        ]

    def get_upcoming(self, hours: int = 24) -> List[Reminder]:
        """Get upcoming reminders within specified hours"""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        return [
            r for r in self.reminders.values()
            if not r.completed and now <= r.remind_at <= cutoff
        ]

    def mark_complete(self, reminder_id: str):
        """Mark reminder as complete"""
        if reminder_id in self.reminders:
            self.reminders[reminder_id].completed = True
            self._save()

    def mark_spoken(self, reminder_id: str):
        """Mark reminder as spoken"""
        if reminder_id in self.reminders:
            self.reminders[reminder_id].spoken = True
            self._save()

    def start_checker(self, check_interval: int = 30):
        """Start background reminder checker"""
        if self.running:
            return

        self.running = True

        def checker_loop():
            while self.running:
                try:
                    pending = self.get_pending()
                    for reminder in pending:
                        if not reminder.spoken and self.on_reminder:
                            self.on_reminder(reminder)
                            self.mark_spoken(reminder.id)
                except Exception as e:
                    logger.error(f"Reminder check error: {e}")
                time.sleep(check_interval)

        self._thread = threading.Thread(target=checker_loop, daemon=True)
        self._thread.start()
        logger.info("Reminder checker started")

    def stop_checker(self):
        """Stop reminder checker"""
        self.running = False

    def parse_reminder_request(self, text: str) -> Dict[str, Any]:
        """Parse natural language reminder request"""
        import re

        text_lower = text.lower()

        # Extract message
        message = text

        # Remove common prefixes
        for prefix in ["remind me to ", "remind me ", "set a reminder to ", "set reminder "]:
            if text_lower.startswith(prefix):
                message = text[len(prefix):]
                break

        # Extract time
        remind_at = None
        delay_minutes = None

        # "in X minutes/hours"
        in_match = re.search(r'in (\d+)\s*(minute|min|hour|hr|second|sec)s?', text_lower)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)

            if 'min' in unit:
                delay_minutes = amount
            elif 'hour' in unit or 'hr' in unit:
                delay_minutes = amount * 60
            elif 'sec' in unit:
                delay_minutes = amount / 60

            # Remove time part from message
            message = re.sub(r'\s*in \d+\s*(minute|min|hour|hr|second|sec)s?', '', message, flags=re.I)

        # "at X:XX" or "at X pm"
        at_match = re.search(r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text_lower)
        if at_match:
            hour = int(at_match.group(1))
            minute = int(at_match.group(2)) if at_match.group(2) else 0
            ampm = at_match.group(3)

            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0

            remind_at = datetime.now().replace(hour=hour, minute=minute, second=0)
            if remind_at < datetime.now():
                remind_at += timedelta(days=1)

            message = re.sub(r'\s*at \d{1,2}(?::\d{2})?\s*(am|pm)?', '', message, flags=re.I)

        # "tomorrow"
        if 'tomorrow' in text_lower:
            base_time = datetime.now() + timedelta(days=1)
            if remind_at:
                remind_at = remind_at.replace(year=base_time.year, month=base_time.month, day=base_time.day)
            else:
                remind_at = base_time.replace(hour=9, minute=0, second=0)
            message = message.replace('tomorrow', '').strip()

        return {
            "message": message.strip(),
            "remind_at": remind_at,
            "delay_minutes": delay_minutes
        }

    def get_summary(self) -> str:
        """Get reminder summary"""
        pending = [r for r in self.reminders.values() if not r.completed]

        if not pending:
            return "You have no pending reminders."

        # Sort by time
        pending.sort(key=lambda r: r.remind_at)

        lines = [f"You have {len(pending)} reminder(s):"]
        for r in pending[:5]:  # Show first 5
            time_str = r.remind_at.strftime("%I:%M %p")
            if r.remind_at.date() != datetime.now().date():
                time_str = r.remind_at.strftime("%b %d at %I:%M %p")
            lines.append(f"  - {r.message} ({time_str})")

        if len(pending) > 5:
            lines.append(f"  ... and {len(pending) - 5} more")

        return "\n".join(lines)


# ============================================================================
# GOOGLE CALENDAR INTEGRATION
# ============================================================================

class GoogleCalendarIntegration:
    """
    Google Calendar integration.
    Requires OAuth2 credentials from Google Cloud Console.
    """

    def __init__(self, credentials_path: str = None, token_path: str = None):
        self.credentials_path = credentials_path or os.path.expanduser("~/.jarvis_calendar_credentials.json")
        self.token_path = token_path or os.path.expanduser("~/.jarvis_calendar_token.pickle")
        self.service = None
        self.authenticated = False

    def authenticate(self) -> bool:
        """Authenticate with Google Calendar API"""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = ['https://www.googleapis.com/auth/calendar']  # Full access

            creds = None

            # Load existing token
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif os.path.exists(self.credentials_path):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)

                    with open(self.token_path, 'wb') as token:
                        pickle.dump(creds, token)
                else:
                    logger.warning("No Google Calendar credentials found")
                    return False

            self.service = build('calendar', 'v3', credentials=creds)
            self.authenticated = True
            logger.info("Google Calendar authenticated")
            return True

        except ImportError:
            logger.warning("Google API libraries not installed. Run: pip install google-api-python-client google-auth-oauthlib")
            return False
        except Exception as e:
            logger.error(f"Calendar authentication failed: {e}")
            return False

    def get_upcoming_events(self, max_results: int = 10,
                            time_range_hours: int = 24) -> List[Dict]:
        """Get upcoming calendar events"""
        if not self.authenticated:
            if not self.authenticate():
                return []

        try:
            now = datetime.utcnow().isoformat() + 'Z'
            end = (datetime.utcnow() + timedelta(hours=time_range_hours)).isoformat() + 'Z'

            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=end,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            return [
                {
                    "summary": e.get("summary", "No title"),
                    "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date")),
                    "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date")),
                    "location": e.get("location", ""),
                    "description": e.get("description", "")[:100] if e.get("description") else ""
                }
                for e in events
            ]

        except Exception as e:
            logger.error(f"Failed to get calendar events: {e}")
            return []

    def get_today_events(self) -> List[Dict]:
        """Get today's events"""
        return self.get_upcoming_events(time_range_hours=24)

    def get_week_events(self) -> List[Dict]:
        """Get this week's events (next 7 days)"""
        return self.get_upcoming_events(max_results=20, time_range_hours=168)

    def get_events_for_date(self, target_date: datetime) -> List[Dict]:
        """Get events for a specific date"""
        if not self.authenticated:
            if not self.authenticate():
                return []

        try:
            # Start of target day
            start = target_date.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            # End of target day
            end = target_date.replace(hour=23, minute=59, second=59).isoformat() + 'Z'

            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start,
                timeMax=end,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            return [
                {
                    "summary": e.get("summary", "No title"),
                    "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date")),
                    "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date")),
                    "location": e.get("location", ""),
                    "description": e.get("description", "")[:100] if e.get("description") else ""
                }
                for e in events
            ]

        except Exception as e:
            logger.error(f"Failed to get events for date: {e}")
            return []

    def create_event(self, summary: str, start_time: datetime,
                     end_time: datetime = None, description: str = "",
                     location: str = "", calendar_id: str = 'primary') -> Dict:
        """
        Create a new calendar event.

        Args:
            summary: Event title
            start_time: Start datetime
            end_time: End datetime (defaults to 1 hour after start)
            description: Event description
            location: Event location
            calendar_id: Calendar to add event to

        Returns:
            Created event details or error dict
        """
        if not self.authenticated:
            if not self.authenticate():
                return {"error": "Not authenticated with Google Calendar"}

        try:
            # Default end time is 1 hour after start
            if end_time is None:
                end_time = start_time + timedelta(hours=1)

            event = {
                'summary': summary,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'America/Los_Angeles',  # Adjust as needed
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
            }

            if description:
                event['description'] = description
            if location:
                event['location'] = location

            created_event = self.service.events().insert(
                calendarId=calendar_id,
                body=event
            ).execute()

            logger.info(f"Created event: {summary} at {start_time}")

            return {
                "success": True,
                "id": created_event.get('id'),
                "summary": created_event.get('summary'),
                "start": created_event.get('start', {}).get('dateTime'),
                "end": created_event.get('end', {}).get('dateTime'),
                "link": created_event.get('htmlLink')
            }

        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return {"error": str(e)}

    def parse_event_request(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language event creation request.

        Examples:
            "schedule a meeting tomorrow at 2pm"
            "add dentist appointment on Friday at 10am"
            "create event lunch with Sarah at noon for 2 hours"
        """
        import re

        text_lower = text.lower()

        # Extract event title (what comes after key phrases)
        summary = text
        for prefix in ["schedule ", "add ", "create event ", "create ",
                       "put ", "book ", "set up "]:
            if text_lower.startswith(prefix):
                summary = text[len(prefix):]
                break

        # Default values
        start_time = None
        duration_hours = 1
        location = ""

        # Extract duration "for X hours/minutes"
        duration_match = re.search(r'for (\d+)\s*(hour|hr|minute|min)s?', text_lower)
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2)
            if 'min' in unit:
                duration_hours = amount / 60
            else:
                duration_hours = amount
            summary = re.sub(r'\s*for \d+\s*(hour|hr|minute|min)s?', '', summary, flags=re.I)

        # Extract location "at [place]" (different from time "at")
        location_match = re.search(r'at (?!(\d|noon|midnight))([^,]+?)(?:\s+(?:on|at|tomorrow|today|this)|\s*$)', text_lower)
        if location_match:
            location = location_match.group(2).strip()

        # Extract time "at X:XX" or "at X pm" or "at noon"
        time_hour = None
        time_minute = 0

        if 'at noon' in text_lower:
            time_hour = 12
            time_minute = 0
            summary = re.sub(r'\s*at noon', '', summary, flags=re.I)
        elif 'at midnight' in text_lower:
            time_hour = 0
            time_minute = 0
            summary = re.sub(r'\s*at midnight', '', summary, flags=re.I)
        else:
            at_match = re.search(r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text_lower)
            if at_match:
                time_hour = int(at_match.group(1))
                time_minute = int(at_match.group(2)) if at_match.group(2) else 0
                ampm = at_match.group(3)

                if ampm == 'pm' and time_hour < 12:
                    time_hour += 12
                elif ampm == 'am' and time_hour == 12:
                    time_hour = 0

                summary = re.sub(r'\s*at \d{1,2}(?::\d{2})?\s*(am|pm)?', '', summary, flags=re.I)

        # Extract date
        base_date = datetime.now()

        if 'tomorrow' in text_lower:
            base_date = datetime.now() + timedelta(days=1)
            summary = summary.replace('tomorrow', '').strip()
        elif 'today' in text_lower:
            base_date = datetime.now()
            summary = summary.replace('today', '').strip()
        else:
            # Check for day names
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for i, day in enumerate(days):
                if day in text_lower:
                    current_day = datetime.now().weekday()
                    days_ahead = i - current_day
                    if days_ahead <= 0:
                        days_ahead += 7
                    base_date = datetime.now() + timedelta(days=days_ahead)
                    summary = re.sub(rf'\s*(?:on\s+)?{day}', '', summary, flags=re.I)
                    break

        # Build start time
        if time_hour is not None:
            start_time = base_date.replace(hour=time_hour, minute=time_minute, second=0, microsecond=0)
        else:
            # Default to 9am if no time specified
            start_time = base_date.replace(hour=9, minute=0, second=0, microsecond=0)

        # Make sure start time is in the future
        if start_time < datetime.now():
            if time_hour is not None:
                start_time += timedelta(days=1)

        # Clean up summary
        summary = re.sub(r'\s+', ' ', summary).strip()
        # Remove trailing "on" or "at"
        summary = re.sub(r'\s+(on|at)$', '', summary, flags=re.I)

        end_time = start_time + timedelta(hours=duration_hours)

        return {
            "summary": summary,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
            "duration_hours": duration_hours
        }

    def get_summary(self) -> str:
        """Get calendar summary for voice"""
        if not self.authenticated:
            return "Calendar not connected. Please set up Google Calendar integration."

        events = self.get_today_events()

        if not events:
            return "You have no events scheduled for today."

        lines = [f"You have {len(events)} event(s) today:"]
        for e in events:
            try:
                start = datetime.fromisoformat(e["start"].replace("Z", "+00:00"))
                time_str = start.strftime("%I:%M %p")
            except:
                time_str = "All day"

            lines.append(f"  - {e['summary']} at {time_str}")
            if e.get("location"):
                lines.append(f"    Location: {e['location']}")

        return "\n".join(lines)


# ============================================================================
# EMAIL INTEGRATION
# ============================================================================

class EmailIntegration:
    """
    Email integration using IMAP.
    Supports Gmail, Outlook, and other IMAP servers.
    """

    def __init__(self):
        self.config_path = os.path.expanduser("~/.jarvis_email_config.json")
        self.config = {}
        self.connected = False

        self._load_config()

    def _load_config(self):
        """Load email configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load email config: {e}")

    def configure(self, email: str, password: str, imap_server: str = None,
                  imap_port: int = 993):
        """Configure email settings"""
        # Auto-detect IMAP server
        if not imap_server:
            domain = email.split('@')[1].lower()
            servers = {
                'gmail.com': 'imap.gmail.com',
                'outlook.com': 'imap-mail.outlook.com',
                'hotmail.com': 'imap-mail.outlook.com',
                'yahoo.com': 'imap.mail.yahoo.com',
            }
            imap_server = servers.get(domain, f'imap.{domain}')

        self.config = {
            'email': email,
            'password': password,
            'imap_server': imap_server,
            'imap_port': imap_port
        }

        # Save config (note: storing passwords is not secure, use app passwords)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f)
            os.chmod(self.config_path, 0o600)  # Restrict permissions
        except Exception as e:
            logger.error(f"Could not save email config: {e}")

    def get_recent_emails(self, count: int = 10, unread_only: bool = True) -> List[Dict]:
        """Get recent emails"""
        if not self.config:
            return []

        try:
            import imaplib
            import email
            from email.header import decode_header

            # Connect to IMAP server
            mail = imaplib.IMAP4_SSL(
                self.config['imap_server'],
                self.config['imap_port']
            )
            mail.login(self.config['email'], self.config['password'])
            mail.select('inbox')

            # Search for emails
            search_criteria = 'UNSEEN' if unread_only else 'ALL'
            status, messages = mail.search(None, search_criteria)

            if status != 'OK':
                return []

            email_ids = messages[0].split()
            emails = []

            # Get last N emails
            for email_id in email_ids[-count:]:
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                if status != 'OK':
                    continue

                msg = email.message_from_bytes(msg_data[0][1])

                # Decode subject
                subject, encoding = decode_header(msg['Subject'])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or 'utf-8')

                # Decode sender
                sender, encoding = decode_header(msg['From'])[0]
                if isinstance(sender, bytes):
                    sender = sender.decode(encoding or 'utf-8')

                # Get date
                date_str = msg['Date']

                emails.append({
                    'id': email_id.decode(),
                    'subject': subject,
                    'sender': sender,
                    'date': date_str,
                    'preview': self._get_email_preview(msg)
                })

            mail.logout()
            return list(reversed(emails))  # Newest first

        except ImportError:
            logger.warning("Email libraries not available")
            return []
        except Exception as e:
            logger.error(f"Email fetch error: {e}")
            return []

    def _get_email_preview(self, msg, max_length: int = 200) -> str:
        """Get email body preview"""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode()
                        return body[:max_length].strip()
            else:
                body = msg.get_payload(decode=True).decode()
                return body[:max_length].strip()
        except:
            return ""

    def get_summary(self) -> str:
        """Get email summary for voice"""
        if not self.config:
            return "Email not configured. Please set up email integration."

        try:
            emails = self.get_recent_emails(count=5, unread_only=True)

            if not emails:
                return "You have no unread emails."

            lines = [f"You have {len(emails)} unread email(s):"]
            for e in emails[:5]:
                sender = e['sender'].split('<')[0].strip()
                lines.append(f"  - From {sender}: {e['subject']}")

            return "\n".join(lines)

        except Exception as e:
            return f"Could not check emails: {str(e)}"


# ============================================================================
# UNIFIED CALENDAR/REMINDER MANAGER
# ============================================================================

class JarvisScheduler:
    """
    Unified scheduler combining calendar, email, and reminders.
    """

    def __init__(self):
        self.reminders = ReminderManager()
        self.calendar = GoogleCalendarIntegration()
        self.email = EmailIntegration()

        # Callbacks
        self.speak: Optional[callable] = None

        # Auto-authenticate calendar on init
        try:
            self.calendar.authenticate()
        except Exception as e:
            logger.warning(f"Calendar auto-auth failed: {e}")

    def start(self):
        """Start scheduler services"""
        # Set up reminder callback
        def on_reminder(reminder):
            if self.speak:
                self.speak(f"Reminder: {reminder.message}")
            logger.info(f"Reminder triggered: {reminder.message}")

        self.reminders.on_reminder = on_reminder
        self.reminders.start_checker()

        # Try to authenticate calendar (silent fail if not configured)
        try:
            self.calendar.authenticate()
        except:
            pass

    def stop(self):
        """Stop scheduler"""
        self.reminders.stop_checker()

    def add_reminder(self, text: str) -> str:
        """Add reminder from natural language"""
        parsed = self.reminders.parse_reminder_request(text)

        reminder = self.reminders.add_reminder(
            message=parsed["message"],
            remind_at=parsed["remind_at"],
            delay_minutes=parsed["delay_minutes"]
        )

        time_str = reminder.remind_at.strftime("%I:%M %p")
        if reminder.remind_at.date() != datetime.now().date():
            time_str = reminder.remind_at.strftime("%B %d at %I:%M %p")

        return f"I'll remind you to {reminder.message} at {time_str}"

    def add_event(self, text: str) -> str:
        """Add calendar event from natural language"""
        if not self.calendar.authenticated:
            return "Calendar not connected. Please set up Google Calendar integration first."

        # Parse the request
        parsed = self.calendar.parse_event_request(text)

        # Create the event
        result = self.calendar.create_event(
            summary=parsed["summary"],
            start_time=parsed["start_time"],
            end_time=parsed["end_time"],
            location=parsed.get("location", "")
        )

        if "error" in result:
            return f"Sorry, I couldn't create the event: {result['error']}"

        # Format response
        start_time = parsed["start_time"]
        time_str = start_time.strftime("%I:%M %p")
        date_str = start_time.strftime("%A, %B %d")

        if start_time.date() == datetime.now().date():
            date_str = "today"
        elif start_time.date() == (datetime.now() + timedelta(days=1)).date():
            date_str = "tomorrow"

        response = f"I've added '{parsed['summary']}' to your calendar for {date_str} at {time_str}"
        if parsed.get("location"):
            response += f" at {parsed['location']}"

        return response

    def get_schedule(self) -> str:
        """Get today's schedule (calendar + reminders)"""
        parts = []

        # Calendar events
        if self.calendar.authenticated:
            events = self.calendar.get_today_events()
            if events:
                parts.append("Calendar events:")
                for e in events[:3]:
                    parts.append(f"  - {e['summary']}")

        # Reminders
        upcoming = self.reminders.get_upcoming(hours=24)
        if upcoming:
            parts.append("Reminders:")
            for r in upcoming[:3]:
                time_str = r.remind_at.strftime("%I:%M %p")
                parts.append(f"  - {r.message} at {time_str}")

        if not parts:
            return "Your schedule is clear for today."

        return "\n".join(parts)

    def get_day_schedule(self, day_name: str) -> str:
        """Get schedule for a specific day"""
        import re

        # Parse the day name to get target date
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_lower = day_name.lower()

        target_date = None

        # Check for "today" or "tomorrow"
        if 'today' in day_lower:
            target_date = datetime.now()
        elif 'tomorrow' in day_lower:
            target_date = datetime.now() + timedelta(days=1)
        else:
            # Find day name
            for i, day in enumerate(days):
                if day in day_lower:
                    current_day = datetime.now().weekday()
                    days_ahead = i - current_day
                    if days_ahead < 0:
                        days_ahead += 7
                    target_date = datetime.now() + timedelta(days=days_ahead)
                    break

        # Check for specific date like "30th" or "december 30"
        date_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', day_lower)
        if date_match:
            day_num = int(date_match.group(1))
            # Assume current or next month
            now = datetime.now()
            try:
                target_date = now.replace(day=day_num)
                if target_date < now:
                    # Try next month
                    if now.month == 12:
                        target_date = now.replace(year=now.year+1, month=1, day=day_num)
                    else:
                        target_date = now.replace(month=now.month+1, day=day_num)
            except ValueError:
                pass

        if not target_date:
            return "I couldn't understand which day you meant. Try 'Tuesday' or 'December 30th'."

        if not self.calendar.authenticated:
            return "Calendar not connected."

        events = self.calendar.get_events_for_date(target_date)
        date_str = target_date.strftime("%A, %B %d")

        if not events:
            return f"No events scheduled for {date_str}."

        parts = [f"Events for {date_str}:"]
        for e in events:
            try:
                start = datetime.fromisoformat(e["start"].replace("Z", "+00:00"))
                time_str = start.strftime("%I:%M %p")
            except:
                time_str = "All day"
            parts.append(f"  - {time_str}: {e['summary']}")

        return "\n".join(parts)

    def get_week_schedule(self) -> str:
        """Get this week's schedule (calendar + reminders)"""
        parts = []

        # Calendar events for the week
        if self.calendar.authenticated:
            events = self.calendar.get_week_events()
            if events:
                parts.append("Calendar events this week:")
                for e in events[:10]:
                    try:
                        start = datetime.fromisoformat(e["start"].replace("Z", "+00:00"))
                        date_str = start.strftime("%A, %B %d at %I:%M %p")
                    except:
                        date_str = e.get("start", "")
                    parts.append(f"  - {date_str}: {e['summary']}")

        # Reminders for the week
        upcoming = self.reminders.get_upcoming(hours=168)
        if upcoming:
            parts.append("\nReminders this week:")
            for r in upcoming[:5]:
                time_str = r.remind_at.strftime("%A, %B %d at %I:%M %p")
                parts.append(f"  - {r.message} at {time_str}")

        if not parts:
            return "Your schedule is clear for this week."

        return "\n".join(parts)

    def get_morning_briefing(self) -> str:
        """Get full morning briefing"""
        parts = []

        # Time and date
        now = datetime.now()
        parts.append(f"Good morning! It's {now.strftime('%A, %B %d')}.")

        # Calendar
        if self.calendar.authenticated:
            events = self.calendar.get_today_events()
            if events:
                parts.append(f"\nYou have {len(events)} event(s) today:")
                for e in events[:3]:
                    parts.append(f"  - {e['summary']}")
            else:
                parts.append("\nNo calendar events today.")

        # Reminders
        reminders = self.reminders.get_upcoming(hours=24)
        if reminders:
            parts.append(f"\n{len(reminders)} reminder(s) for today.")

        # Unread emails
        if self.email.config:
            try:
                emails = self.email.get_recent_emails(count=5, unread_only=True)
                if emails:
                    parts.append(f"\nYou have {len(emails)} unread email(s).")
            except:
                pass

        return "\n".join(parts)


# Create global instance
scheduler = JarvisScheduler()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Calendar & Reminder System...")

    # Test reminders
    print("\n=== Reminders ===")
    r = scheduler.reminders.add_reminder("Test reminder", delay_minutes=5)
    print(f"Added: {r.message} at {r.remind_at}")

    # Test parsing
    print("\n=== Parsing Tests ===")
    tests = [
        "remind me to call mom in 30 minutes",
        "set a reminder to check the oven at 6pm",
        "remind me tomorrow to pay bills",
    ]

    for test in tests:
        parsed = scheduler.reminders.parse_reminder_request(test)
        print(f"  '{test}'")
        print(f"    -> message: {parsed['message']}")
        print(f"    -> at: {parsed['remind_at'] or f'in {parsed['delay_minutes']} min'}")

    print("\n=== Summary ===")
    print(scheduler.reminders.get_summary())

    print("\n=== Schedule ===")
    print(scheduler.get_schedule())
