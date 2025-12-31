#!/usr/bin/env python3
"""
JARVIS Proactive Systems
- System Monitoring (GPU, CPU, Memory, Disk)
- Training Monitor
- Proactive Alerts
- Task Automation/Routines
- Ambient Awareness
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger("JarvisProactive")

# ============================================================================
# SYSTEM MONITORING
# ============================================================================

@dataclass
class SystemStats:
    """Current system statistics"""
    timestamp: datetime
    gpu_usage: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_temp: float = 0.0
    cpu_usage: float = 0.0
    memory_used: float = 0.0
    memory_total: float = 0.0
    disk_used: float = 0.0
    disk_total: float = 0.0


class SystemMonitor:
    """Monitor system resources in real-time"""

    def __init__(self):
        self.last_stats: Optional[SystemStats] = None
        self.history: List[SystemStats] = []
        self.max_history = 3600  # 1 hour at 1 sample/sec
        self.running = False
        self._thread = None

        # Alert thresholds
        self.thresholds = {
            "gpu_temp": 80,  # Celsius
            "gpu_memory": 95,  # Percent
            "cpu_usage": 90,  # Percent
            "memory": 90,  # Percent
            "disk": 90,  # Percent
        }

        # Callbacks for alerts
        self.on_alert: Optional[Callable[[str, str], None]] = None

    def get_gpu_stats(self) -> Dict[str, float]:
        """Get NVIDIA GPU statistics"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 4:
                    return {
                        "usage": float(parts[0]),
                        "memory_used": float(parts[1]) / 1024,  # GB
                        "memory_total": float(parts[2]) / 1024,  # GB
                        "temp": float(parts[3])
                    }
        except Exception as e:
            logger.debug(f"GPU stats error: {e}")
        return {"usage": 0, "memory_used": 0, "memory_total": 0, "temp": 0}

    def get_cpu_stats(self) -> float:
        """Get CPU usage percentage"""
        try:
            # Use /proc/stat for CPU usage
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = line.split()
            idle = float(parts[4])
            total = sum(float(p) for p in parts[1:])

            if not hasattr(self, '_last_cpu'):
                self._last_cpu = (idle, total)
                return 0.0

            last_idle, last_total = self._last_cpu
            idle_delta = idle - last_idle
            total_delta = total - last_total
            self._last_cpu = (idle, total)

            if total_delta > 0:
                return 100.0 * (1.0 - idle_delta / total_delta)
        except Exception as e:
            logger.debug(f"CPU stats error: {e}")
        return 0.0

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()

            mem_info = {}
            for line in lines:
                parts = line.split()
                key = parts[0].rstrip(':')
                value = float(parts[1]) / 1024 / 1024  # KB to GB
                mem_info[key] = value

            total = mem_info.get('MemTotal', 0)
            available = mem_info.get('MemAvailable', 0)
            used = total - available

            return {"used": used, "total": total}
        except Exception as e:
            logger.debug(f"Memory stats error: {e}")
        return {"used": 0, "total": 0}

    def get_disk_stats(self, path: str = "/") -> Dict[str, float]:
        """Get disk usage in GB"""
        try:
            stat = os.statvfs(path)
            total = (stat.f_blocks * stat.f_frsize) / (1024**3)
            free = (stat.f_bavail * stat.f_frsize) / (1024**3)
            used = total - free
            return {"used": used, "total": total}
        except Exception as e:
            logger.debug(f"Disk stats error: {e}")
        return {"used": 0, "total": 0}

    def collect_stats(self) -> SystemStats:
        """Collect all system statistics"""
        gpu = self.get_gpu_stats()
        memory = self.get_memory_stats()
        disk = self.get_disk_stats()

        stats = SystemStats(
            timestamp=datetime.now(),
            gpu_usage=gpu["usage"],
            gpu_memory_used=gpu["memory_used"],
            gpu_memory_total=gpu["memory_total"],
            gpu_temp=gpu["temp"],
            cpu_usage=self.get_cpu_stats(),
            memory_used=memory["used"],
            memory_total=memory["total"],
            disk_used=disk["used"],
            disk_total=disk["total"]
        )

        self.last_stats = stats
        self.history.append(stats)

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Check for alerts
        self._check_alerts(stats)

        return stats

    def _check_alerts(self, stats: SystemStats):
        """Check if any thresholds are exceeded"""
        if not self.on_alert:
            return

        if stats.gpu_temp > self.thresholds["gpu_temp"]:
            self.on_alert("gpu_temp", f"GPU temperature is {stats.gpu_temp:.0f}°C - consider cooling")

        gpu_mem_pct = (stats.gpu_memory_used / stats.gpu_memory_total * 100) if stats.gpu_memory_total > 0 else 0
        if gpu_mem_pct > self.thresholds["gpu_memory"]:
            self.on_alert("gpu_memory", f"GPU memory at {gpu_mem_pct:.0f}% - {stats.gpu_memory_used:.1f}/{stats.gpu_memory_total:.1f} GB")

        if stats.cpu_usage > self.thresholds["cpu_usage"]:
            self.on_alert("cpu_usage", f"CPU usage at {stats.cpu_usage:.0f}%")

        mem_pct = (stats.memory_used / stats.memory_total * 100) if stats.memory_total > 0 else 0
        if mem_pct > self.thresholds["memory"]:
            self.on_alert("memory", f"System memory at {mem_pct:.0f}%")

    def start(self, interval: float = 5.0):
        """Start background monitoring"""
        if self.running:
            return

        self.running = True

        def monitor_loop():
            while self.running:
                try:
                    self.collect_stats()
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                time.sleep(interval)

        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        logger.info("System monitoring started")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("System monitoring stopped")

    def get_summary(self) -> str:
        """Get human-readable system summary"""
        if not self.last_stats:
            self.collect_stats()

        s = self.last_stats
        gpu_mem_pct = (s.gpu_memory_used / s.gpu_memory_total * 100) if s.gpu_memory_total > 0 else 0
        mem_pct = (s.memory_used / s.memory_total * 100) if s.memory_total > 0 else 0
        disk_pct = (s.disk_used / s.disk_total * 100) if s.disk_total > 0 else 0

        return (
            f"GPU: {s.gpu_usage:.0f}% usage, {s.gpu_memory_used:.1f}/{s.gpu_memory_total:.0f} GB ({gpu_mem_pct:.0f}%), {s.gpu_temp:.0f}°C\n"
            f"CPU: {s.cpu_usage:.0f}% usage\n"
            f"Memory: {s.memory_used:.1f}/{s.memory_total:.0f} GB ({mem_pct:.0f}%)\n"
            f"Disk: {s.disk_used:.0f}/{s.disk_total:.0f} GB ({disk_pct:.0f}%)"
        )


# ============================================================================
# TRAINING MONITOR
# ============================================================================

@dataclass
class TrainingStatus:
    """Training job status"""
    job_name: str
    status: str  # running, completed, failed, unknown
    progress: float  # 0-100
    current_step: int
    total_steps: int
    current_loss: float
    best_loss: float
    eta: Optional[str]
    start_time: Optional[datetime]
    last_update: datetime


class TrainingMonitor:
    """Monitor AI training jobs"""

    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}
        self.log_watchers: Dict[str, threading.Thread] = {}
        self.running = False

        # Callbacks
        self.on_progress: Optional[Callable[[str, TrainingStatus], None]] = None
        self.on_complete: Optional[Callable[[str, TrainingStatus], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None

    def watch_log(self, job_name: str, log_path: str):
        """Watch a training log file for updates"""

        def watcher():
            last_size = 0

            while self.running and job_name in self.log_watchers:
                try:
                    if os.path.exists(log_path):
                        size = os.path.getsize(log_path)
                        if size > last_size:
                            with open(log_path, 'r') as f:
                                f.seek(last_size)
                                new_content = f.read()
                                self._parse_log_update(job_name, new_content)
                            last_size = size
                except Exception as e:
                    logger.debug(f"Log watch error: {e}")

                time.sleep(2)

        self.log_watchers[job_name] = threading.Thread(target=watcher, daemon=True)
        self.log_watchers[job_name].start()

    def _parse_log_update(self, job_name: str, content: str):
        """Parse training log content for status updates"""
        # Look for common patterns
        import re

        # Step/Epoch progress
        step_match = re.search(r'(?:step|iter|iteration)[:\s]+(\d+)[/\s]+(\d+)', content, re.I)
        epoch_match = re.search(r'epoch[:\s]+(\d+)[/\s]+(\d+)', content, re.I)

        # Loss values
        loss_match = re.search(r'(?:loss|train_loss)[:\s]+([\d.]+)', content, re.I)

        # Progress percentage
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', content)

        if job_name not in self.jobs:
            self.jobs[job_name] = TrainingStatus(
                job_name=job_name,
                status="running",
                progress=0,
                current_step=0,
                total_steps=0,
                current_loss=0,
                best_loss=float('inf'),
                eta=None,
                start_time=datetime.now(),
                last_update=datetime.now()
            )

        status = self.jobs[job_name]

        if step_match:
            status.current_step = int(step_match.group(1))
            status.total_steps = int(step_match.group(2))
            if status.total_steps > 0:
                status.progress = (status.current_step / status.total_steps) * 100

        if loss_match:
            status.current_loss = float(loss_match.group(1))
            if status.current_loss < status.best_loss:
                status.best_loss = status.current_loss

        if pct_match:
            status.progress = float(pct_match.group(1))

        status.last_update = datetime.now()

        # Check for completion
        if "completed" in content.lower() or "finished" in content.lower():
            status.status = "completed"
            if self.on_complete:
                self.on_complete(job_name, status)

        # Check for errors
        if "error" in content.lower() or "failed" in content.lower():
            status.status = "failed"
            if self.on_error:
                self.on_error(job_name, content[-500:])

        if self.on_progress:
            self.on_progress(job_name, status)

    def check_docker_training(self, container: str = "alexandra_gpu") -> Optional[TrainingStatus]:
        """Check training status in Docker container"""
        try:
            # Look for common training processes
            result = subprocess.run(
                ["docker", "exec", container, "pgrep", "-a", "-f", "train|accelerate|python.*train"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                processes = result.stdout.strip().split('\n')
                return TrainingStatus(
                    job_name="docker_training",
                    status="running",
                    progress=0,
                    current_step=0,
                    total_steps=0,
                    current_loss=0,
                    best_loss=0,
                    eta=None,
                    start_time=None,
                    last_update=datetime.now()
                )
        except Exception as e:
            logger.debug(f"Docker check error: {e}")

        return None

    def get_status_summary(self) -> str:
        """Get summary of all training jobs"""
        if not self.jobs:
            docker_status = self.check_docker_training()
            if docker_status:
                return "Training job detected in Docker container (no log attached)"
            return "No active training jobs"

        summaries = []
        for name, status in self.jobs.items():
            summary = f"{name}: {status.status}"
            if status.progress > 0:
                summary += f" - {status.progress:.1f}%"
            if status.current_loss > 0:
                summary += f" - Loss: {status.current_loss:.4f}"
            if status.eta:
                summary += f" - ETA: {status.eta}"
            summaries.append(summary)

        return "\n".join(summaries)

    def start(self):
        """Start training monitor"""
        self.running = True
        logger.info("Training monitor started")

    def stop(self):
        """Stop training monitor"""
        self.running = False
        self.log_watchers.clear()
        logger.info("Training monitor stopped")


# ============================================================================
# TASK AUTOMATION / ROUTINES
# ============================================================================

@dataclass
class RoutineStep:
    """Single step in a routine"""
    action: str  # Type: speak, run_command, open_app, open_url, wait, get_info
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None  # Optional condition to check


@dataclass
class Routine:
    """Automated routine/macro"""
    name: str
    description: str
    steps: List[RoutineStep]
    trigger: Optional[str] = None  # voice trigger phrase
    schedule: Optional[str] = None  # cron-like schedule


class RoutineManager:
    """Manage and execute automated routines"""

    def __init__(self):
        self.routines: Dict[str, Routine] = {}
        self.running_routines: Dict[str, bool] = {}

        # Callbacks for actions
        self.speak: Optional[Callable[[str], None]] = None
        self.run_command: Optional[Callable[[str], str]] = None
        self.get_weather: Optional[Callable[[], str]] = None
        self.get_news: Optional[Callable[[], str]] = None
        self.get_time: Optional[Callable[[], str]] = None
        self.get_system_status: Optional[Callable[[], str]] = None

        # Create default routines
        self._create_default_routines()

    def _create_default_routines(self):
        """Create default routines"""

        # Good morning routine
        self.routines["morning"] = Routine(
            name="morning",
            description="Good morning briefing",
            trigger="good morning",
            steps=[
                RoutineStep("speak", {"text": "Good morning! Let me give you your daily briefing."}),
                RoutineStep("get_info", {"type": "time"}),
                RoutineStep("get_info", {"type": "weather"}),
                RoutineStep("get_info", {"type": "news"}),
                RoutineStep("get_info", {"type": "system"}),
                RoutineStep("speak", {"text": "That's your morning update. Have a great day!"}),
            ]
        )

        # System check routine
        self.routines["system_check"] = Routine(
            name="system_check",
            description="Full system status check",
            trigger="system check",
            steps=[
                RoutineStep("speak", {"text": "Running system diagnostics..."}),
                RoutineStep("get_info", {"type": "system"}),
                RoutineStep("get_info", {"type": "training"}),
                RoutineStep("speak", {"text": "System check complete."}),
            ]
        )

        # Shutdown routine
        self.routines["shutdown"] = Routine(
            name="shutdown",
            description="End of day shutdown routine",
            trigger="shutdown routine",
            steps=[
                RoutineStep("speak", {"text": "Starting shutdown routine..."}),
                RoutineStep("get_info", {"type": "system"}),
                RoutineStep("speak", {"text": "All systems nominal. Goodbye!"}),
            ]
        )

        # Training check routine
        self.routines["training_status"] = Routine(
            name="training_status",
            description="Check training progress",
            trigger="training status",
            steps=[
                RoutineStep("get_info", {"type": "training"}),
            ]
        )

    def add_routine(self, routine: Routine):
        """Add a new routine"""
        self.routines[routine.name] = routine

    def get_routine_by_trigger(self, text: str) -> Optional[Routine]:
        """Find routine matching trigger phrase"""
        text_lower = text.lower()
        for routine in self.routines.values():
            if routine.trigger and routine.trigger.lower() in text_lower:
                return routine
        return None

    async def execute_routine(self, name: str) -> List[str]:
        """Execute a routine and return results"""
        if name not in self.routines:
            return [f"Routine '{name}' not found"]

        routine = self.routines[name]
        results = []

        self.running_routines[name] = True

        try:
            for step in routine.steps:
                if not self.running_routines.get(name, False):
                    results.append("Routine cancelled")
                    break

                result = await self._execute_step(step)
                if result:
                    results.append(result)
        finally:
            self.running_routines[name] = False

        return results

    async def _execute_step(self, step: RoutineStep) -> Optional[str]:
        """Execute a single routine step"""
        action = step.action
        params = step.params

        if action == "speak" and self.speak:
            self.speak(params.get("text", ""))
            return params.get("text")

        elif action == "wait":
            await asyncio.sleep(params.get("seconds", 1))
            return None

        elif action == "run_command" and self.run_command:
            return self.run_command(params.get("command", ""))

        elif action == "get_info":
            info_type = params.get("type", "")

            if info_type == "weather" and self.get_weather:
                weather = self.get_weather()
                if self.speak:
                    self.speak(weather)
                return weather

            elif info_type == "news" and self.get_news:
                news = self.get_news()
                if self.speak:
                    self.speak(news)
                return news

            elif info_type == "time" and self.get_time:
                time_str = self.get_time()
                if self.speak:
                    self.speak(time_str)
                return time_str

            elif info_type == "system" and self.get_system_status:
                status = self.get_system_status()
                if self.speak:
                    self.speak(status)
                return status

        return None

    def cancel_routine(self, name: str):
        """Cancel a running routine"""
        self.running_routines[name] = False

    def list_routines(self) -> List[Dict[str, str]]:
        """List all available routines"""
        return [
            {
                "name": r.name,
                "description": r.description,
                "trigger": r.trigger or "None"
            }
            for r in self.routines.values()
        ]


# ============================================================================
# PROACTIVE ALERTS SYSTEM
# ============================================================================

class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """System alert"""
    id: str
    category: str
    message: str
    priority: AlertPriority
    timestamp: datetime
    acknowledged: bool = False
    data: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manage proactive alerts"""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.max_alerts = 100
        self._alert_counter = 0

        # Cooldowns to prevent spam
        self._cooldowns: Dict[str, datetime] = {}
        self.cooldown_minutes = 5

        # Callbacks
        self.on_alert: Optional[Callable[[Alert], None]] = None
        self.speak_alert: Optional[Callable[[str], None]] = None

    def add_alert(self, category: str, message: str,
                  priority: AlertPriority = AlertPriority.MEDIUM,
                  data: Dict = None, speak: bool = True) -> Optional[Alert]:
        """Add a new alert with cooldown check"""

        # Check cooldown
        cooldown_key = f"{category}:{message[:50]}"
        if cooldown_key in self._cooldowns:
            if datetime.now() - self._cooldowns[cooldown_key] < timedelta(minutes=self.cooldown_minutes):
                return None  # Still in cooldown

        self._alert_counter += 1
        alert = Alert(
            id=f"alert_{self._alert_counter}",
            category=category,
            message=message,
            priority=priority,
            timestamp=datetime.now(),
            data=data or {}
        )

        self.alerts.append(alert)
        self._cooldowns[cooldown_key] = datetime.now()

        # Trim old alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        # Trigger callback
        if self.on_alert:
            self.on_alert(alert)

        # Speak high priority alerts
        if speak and priority.value >= AlertPriority.HIGH.value and self.speak_alert:
            self.speak_alert(message)

        return alert

    def get_unacknowledged(self) -> List[Alert]:
        """Get unacknowledged alerts"""
        return [a for a in self.alerts if not a.acknowledged]

    def acknowledge(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break

    def acknowledge_all(self):
        """Acknowledge all alerts"""
        for alert in self.alerts:
            alert.acknowledged = True

    def get_summary(self) -> str:
        """Get alert summary"""
        unack = self.get_unacknowledged()
        if not unack:
            return "No pending alerts"

        by_priority = {}
        for alert in unack:
            p = alert.priority.name
            if p not in by_priority:
                by_priority[p] = []
            by_priority[p].append(alert.message)

        summary = []
        for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if priority in by_priority:
                summary.append(f"{priority}: {len(by_priority[priority])} alerts")
                for msg in by_priority[priority][:3]:  # Show first 3
                    summary.append(f"  - {msg}")

        return "\n".join(summary)


# ============================================================================
# FACE RECOGNITION
# ============================================================================

class FaceRecognition:
    """Face detection and recognition"""

    def __init__(self):
        self.known_faces: Dict[str, Any] = {}  # name -> encoding
        self.last_seen: Dict[str, datetime] = {}
        self.face_detector = None
        self.face_encoder = None

        # Callbacks
        self.on_recognized: Optional[Callable[[str], None]] = None
        self.on_unknown: Optional[Callable[[], None]] = None

        self._init_models()

    def _init_models(self):
        """Initialize face detection models"""
        try:
            import cv2
            # Use OpenCV's built-in face detector (Haar cascade)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detection initialized with OpenCV Haar cascade")
        except Exception as e:
            logger.warning(f"Face detection init failed: {e}")

    def detect_faces(self, frame) -> List[Dict]:
        """Detect faces in frame"""
        if self.face_detector is None:
            return []

        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            return [
                {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                for (x, y, w, h) in faces
            ]
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []

    def register_face(self, name: str, frame, face_region: Dict):
        """Register a known face"""
        try:
            import cv2
            x, y, w, h = face_region["x"], face_region["y"], face_region["w"], face_region["h"]
            face_img = frame[y:y+h, x:x+w]

            # Store face encoding (simplified - just store the image)
            self.known_faces[name] = cv2.resize(face_img, (128, 128))
            logger.info(f"Registered face: {name}")
            return True
        except Exception as e:
            logger.error(f"Face registration error: {e}")
            return False

    def process_frame(self, frame) -> Dict:
        """Process frame for face detection"""
        faces = self.detect_faces(frame)

        result = {
            "face_count": len(faces),
            "faces": faces,
            "recognized": [],
            "unknown_count": 0
        }

        # Simple recognition based on face detection
        # For full recognition, would need face_recognition library
        if faces:
            for face in faces:
                # Check if we've seen this face location recently
                # This is simplified - real impl would use embeddings
                result["recognized"].append({
                    "name": "User",  # Assume it's the user
                    "confidence": 0.8,
                    "bbox": face
                })

        return result

    def draw_faces(self, frame, faces_info: Dict):
        """Draw face boxes on frame"""
        import cv2

        for face in faces_info.get("recognized", []):
            bbox = face["bbox"]
            name = face["name"]
            conf = face["confidence"]

            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            # Draw box
            color = (0, 255, 0)  # Green for recognized
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw name
            label = f"{name} ({conf:.0%})"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


# ============================================================================
# AMBIENT SOUND DETECTION
# ============================================================================

class AmbientSoundDetector:
    """Detect ambient sounds (doorbell, phone, etc.)"""

    def __init__(self):
        self.running = False
        self._thread = None
        self.audio_stream = None

        # Sound patterns to detect
        self.patterns = {
            "doorbell": {"freq_range": (400, 800), "duration": 0.5},
            "phone": {"freq_range": (800, 1200), "duration": 0.3},
            "alarm": {"freq_range": (1000, 3000), "duration": 0.2},
        }

        # Callbacks
        self.on_sound_detected: Optional[Callable[[str], None]] = None

    def start(self):
        """Start ambient sound detection"""
        # Note: Full implementation would require pyaudio and audio processing
        # This is a placeholder for the architecture
        self.running = True
        logger.info("Ambient sound detection started (placeholder)")

    def stop(self):
        """Stop detection"""
        self.running = False
        logger.info("Ambient sound detection stopped")

    def get_status(self) -> str:
        """Get detection status"""
        if self.running:
            return "Listening for ambient sounds..."
        return "Ambient detection not active"


# ============================================================================
# UNIFIED PROACTIVE SYSTEM
# ============================================================================

class JarvisProactive:
    """
    Unified proactive systems manager.
    Coordinates all monitoring, alerts, and automation.
    """

    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor()
        self.routine_manager = RoutineManager()
        self.alert_manager = AlertManager()
        self.face_recognition = FaceRecognition()
        self.sound_detector = AmbientSoundDetector()

        # Link components
        self.system_monitor.on_alert = self._handle_system_alert
        self.training_monitor.on_complete = self._handle_training_complete
        self.training_monitor.on_error = self._handle_training_error

        # State
        self.running = False

    def start(self):
        """Start all proactive systems"""
        self.running = True
        self.system_monitor.start()
        self.training_monitor.start()
        logger.info("Proactive systems started")

    def stop(self):
        """Stop all systems"""
        self.running = False
        self.system_monitor.stop()
        self.training_monitor.stop()
        self.sound_detector.stop()
        logger.info("Proactive systems stopped")

    def _handle_system_alert(self, category: str, message: str):
        """Handle system monitoring alerts"""
        priority = AlertPriority.HIGH if "temp" in category else AlertPriority.MEDIUM
        self.alert_manager.add_alert(category, message, priority)

    def _handle_training_complete(self, job_name: str, status):
        """Handle training completion"""
        self.alert_manager.add_alert(
            "training",
            f"Training job '{job_name}' completed! Final loss: {status.best_loss:.4f}",
            AlertPriority.HIGH
        )

    def _handle_training_error(self, job_name: str, error: str):
        """Handle training errors"""
        self.alert_manager.add_alert(
            "training_error",
            f"Training job '{job_name}' encountered an error: {error[:100]}",
            AlertPriority.CRITICAL
        )

    def get_full_status(self) -> Dict[str, str]:
        """Get status of all systems"""
        return {
            "system": self.system_monitor.get_summary(),
            "training": self.training_monitor.get_status_summary(),
            "alerts": self.alert_manager.get_summary(),
            "routines": f"{len(self.routine_manager.routines)} routines available"
        }

    def process_camera_frame(self, frame):
        """Process camera frame for face detection"""
        if frame is None:
            return frame, {}

        faces_info = self.face_recognition.process_frame(frame)
        annotated = self.face_recognition.draw_faces(frame.copy(), faces_info)

        return annotated, faces_info


# Create global instance
proactive = JarvisProactive()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("Testing Proactive Systems...")

    # Test system monitor
    print("\n=== System Monitor ===")
    stats = proactive.system_monitor.collect_stats()
    print(proactive.system_monitor.get_summary())

    # Test routines
    print("\n=== Available Routines ===")
    for r in proactive.routine_manager.list_routines():
        print(f"  {r['name']}: {r['description']} (trigger: '{r['trigger']}')")

    # Test alerts
    print("\n=== Alerts ===")
    proactive.alert_manager.add_alert("test", "This is a test alert", AlertPriority.MEDIUM)
    print(proactive.alert_manager.get_summary())

    print("\n=== Full Status ===")
    for key, value in proactive.get_full_status().items():
        print(f"{key}:\n{value}\n")
