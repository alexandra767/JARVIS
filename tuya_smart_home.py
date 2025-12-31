#!/usr/bin/env python3
"""
Tuya Smart Home Integration for Alexandra/Jarvis
Controls Daybetter and other Tuya-compatible smart devices
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import colorsys

logger = logging.getLogger("TuyaSmartHome")

# Try to import tinytuya for local control
try:
    import tinytuya
    TINYTUYA_AVAILABLE = True
except ImportError:
    TINYTUYA_AVAILABLE = False
    logger.warning("tinytuya not installed. Run: pip install tinytuya")


@dataclass
class TuyaDevice:
    """Represents a Tuya device"""
    id: str
    name: str
    device_type: str  # 'light', 'switch', 'plug', etc.
    local_key: str = ""
    ip: str = ""
    version: str = "3.3"

    # State
    is_on: bool = False
    brightness: int = 100
    color_temp: int = 50
    color: tuple = (255, 255, 255)  # RGB


class TuyaSmartHome:
    """
    Control Tuya/Smart Life devices.
    Supports: Daybetter, Gosund, Teckin, and other Tuya-compatible devices.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize Tuya Smart Home controller.

        Config can be loaded from file or set via environment variables:
        - TUYA_ACCESS_ID: Your Tuya Cloud Access ID
        - TUYA_ACCESS_SECRET: Your Tuya Cloud Access Secret
        - TUYA_REGION: Region (us, eu, cn) - default: us
        """
        self.config_path = config_path or os.path.expanduser("~/.alexandra/tuya_config.json")
        self.devices: Dict[str, TuyaDevice] = {}
        self.device_aliases: Dict[str, str] = {}  # Maps room names to device IDs

        # Credentials
        self.access_id = os.environ.get('TUYA_ACCESS_ID', '')
        self.access_secret = os.environ.get('TUYA_ACCESS_SECRET', '')
        self.region = os.environ.get('TUYA_REGION', 'us')

        # Load config if exists
        self._load_config()

        self.available = TINYTUYA_AVAILABLE and bool(self.devices)

    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                self.access_id = config.get('access_id', self.access_id)
                self.access_secret = config.get('access_secret', self.access_secret)
                self.region = config.get('region', self.region)

                # Load devices
                for dev_data in config.get('devices', []):
                    device = TuyaDevice(
                        id=dev_data['id'],
                        name=dev_data['name'],
                        device_type=dev_data.get('type', 'light'),
                        local_key=dev_data.get('local_key', ''),
                        ip=dev_data.get('ip', ''),
                        version=dev_data.get('version', '3.3')
                    )
                    self.devices[device.id] = device

                    # Create aliases from name
                    name_lower = device.name.lower()
                    self.device_aliases[name_lower] = device.id
                    # Also add without "light" suffix
                    if 'light' in name_lower:
                        self.device_aliases[name_lower.replace(' light', '')] = device.id

                logger.info(f"Loaded {len(self.devices)} Tuya devices")

            except Exception as e:
                logger.error(f"Error loading Tuya config: {e}")

    def save_config(self):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        config = {
            'access_id': self.access_id,
            'access_secret': self.access_secret,
            'region': self.region,
            'devices': [
                {
                    'id': d.id,
                    'name': d.name,
                    'type': d.device_type,
                    'local_key': d.local_key,
                    'ip': d.ip,
                    'version': d.version
                }
                for d in self.devices.values()
            ]
        }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved Tuya config to {self.config_path}")

    def add_device(self, device_id: str, name: str, local_key: str,
                   ip: str = "", device_type: str = "light") -> str:
        """Manually add a device"""
        device = TuyaDevice(
            id=device_id,
            name=name,
            device_type=device_type,
            local_key=local_key,
            ip=ip
        )
        self.devices[device_id] = device
        self.device_aliases[name.lower()] = device_id

        self.save_config()
        self.available = True

        return f"Added device: {name} ({device_id})"

    def _get_device_id(self, name_or_id: str) -> Optional[str]:
        """Get device ID from name or ID"""
        # Direct ID match
        if name_or_id in self.devices:
            return name_or_id

        # Alias match
        name_lower = name_or_id.lower().strip()
        if name_lower in self.device_aliases:
            return self.device_aliases[name_lower]

        # Fuzzy match
        for alias, dev_id in self.device_aliases.items():
            if name_lower in alias or alias in name_lower:
                return dev_id

        return None

    def _get_tuya_device(self, device: TuyaDevice):
        """Get tinytuya device object"""
        if not TINYTUYA_AVAILABLE:
            return None

        if device.device_type == 'light':
            d = tinytuya.BulbDevice(
                dev_id=device.id,
                address=device.ip or 'Auto',
                local_key=device.local_key,
                version=float(device.version)
            )
        else:
            d = tinytuya.OutletDevice(
                dev_id=device.id,
                address=device.ip or 'Auto',
                local_key=device.local_key,
                version=float(device.version)
            )

        return d

    # ==================== CONTROL METHODS ====================

    def turn_on(self, device_name: str = "all") -> str:
        """Turn on a light or all lights"""
        if not self.available:
            return self._not_configured_msg()

        if device_name.lower() == "all":
            results = []
            for device in self.devices.values():
                result = self._turn_on_device(device)
                results.append(f"{device.name}: {result}")
            return "Turned on all lights:\n" + "\n".join(results)
        else:
            device_id = self._get_device_id(device_name)
            if not device_id:
                return f"Device not found: {device_name}"
            device = self.devices[device_id]
            result = self._turn_on_device(device)
            return f"Turned on {device.name}: {result}"

    def _turn_on_device(self, device: TuyaDevice) -> str:
        """Turn on a specific device"""
        try:
            d = self._get_tuya_device(device)
            if d:
                d.turn_on()
                device.is_on = True
                return "OK"
            return "Device not available"
        except Exception as e:
            return f"Error: {e}"

    def turn_off(self, device_name: str = "all") -> str:
        """Turn off a light or all lights"""
        if not self.available:
            return self._not_configured_msg()

        if device_name.lower() == "all":
            results = []
            for device in self.devices.values():
                result = self._turn_off_device(device)
                results.append(f"{device.name}: {result}")
            return "Turned off all lights:\n" + "\n".join(results)
        else:
            device_id = self._get_device_id(device_name)
            if not device_id:
                return f"Device not found: {device_name}"
            device = self.devices[device_id]
            result = self._turn_off_device(device)
            return f"Turned off {device.name}: {result}"

    def _turn_off_device(self, device: TuyaDevice) -> str:
        """Turn off a specific device"""
        try:
            d = self._get_tuya_device(device)
            if d:
                d.turn_off()
                device.is_on = False
                return "OK"
            return "Device not available"
        except Exception as e:
            return f"Error: {e}"

    def set_brightness(self, device_name: str, brightness: int) -> str:
        """Set brightness (0-100)"""
        if not self.available:
            return self._not_configured_msg()

        brightness = max(0, min(100, brightness))

        device_id = self._get_device_id(device_name)
        if not device_id:
            return f"Device not found: {device_name}"

        device = self.devices[device_id]

        try:
            d = self._get_tuya_device(device)
            if d and hasattr(d, 'set_brightness_percentage'):
                d.set_brightness_percentage(brightness)
                device.brightness = brightness
                return f"Set {device.name} brightness to {brightness}%"
            return "Device doesn't support brightness"
        except Exception as e:
            return f"Error: {e}"

    def set_color(self, device_name: str, color: str) -> str:
        """Set color by name or hex code"""
        if not self.available:
            return self._not_configured_msg()

        # Parse color
        rgb = self._parse_color(color)
        if not rgb:
            return f"Unknown color: {color}"

        device_id = self._get_device_id(device_name)
        if not device_id:
            # Try all lights
            if device_name.lower() in ['all', 'lights', 'all lights']:
                results = []
                for device in self.devices.values():
                    result = self._set_device_color(device, rgb)
                    results.append(f"{device.name}: {result}")
                return f"Set all lights to {color}:\n" + "\n".join(results)
            return f"Device not found: {device_name}"

        device = self.devices[device_id]
        result = self._set_device_color(device, rgb)
        return f"Set {device.name} to {color}: {result}"

    def _set_device_color(self, device: TuyaDevice, rgb: tuple) -> str:
        """Set device color"""
        try:
            d = self._get_tuya_device(device)
            if d and hasattr(d, 'set_colour'):
                d.set_colour(rgb[0], rgb[1], rgb[2])
                device.color = rgb
                return "OK"
            return "Device doesn't support color"
        except Exception as e:
            return f"Error: {e}"

    def set_white(self, device_name: str, color_temp: int = 50) -> str:
        """Set to white mode with color temperature (0=warm, 100=cool)"""
        if not self.available:
            return self._not_configured_msg()

        device_id = self._get_device_id(device_name)
        if not device_id:
            return f"Device not found: {device_name}"

        device = self.devices[device_id]

        try:
            d = self._get_tuya_device(device)
            if d and hasattr(d, 'set_white_percentage'):
                d.set_white_percentage(brightness=100, colourtemp=color_temp)
                device.color_temp = color_temp
                return f"Set {device.name} to white (temp: {color_temp}%)"
            return "Device doesn't support white mode"
        except Exception as e:
            return f"Error: {e}"

    def _parse_color(self, color: str) -> Optional[tuple]:
        """Parse color name or hex to RGB"""
        color = color.lower().strip()

        # Common color names
        colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'violet': (238, 130, 238),
            'pink': (255, 192, 203),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'warm white': (255, 244, 229),
            'cool white': (255, 255, 255),
            'magenta': (255, 0, 255),
            'lime': (0, 255, 0),
            'teal': (0, 128, 128),
            'coral': (255, 127, 80),
            'gold': (255, 215, 0),
            'lavender': (230, 230, 250),
        }

        if color in colors:
            return colors[color]

        # Hex color
        if color.startswith('#'):
            try:
                hex_color = color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            except:
                pass

        return None

    def list_devices(self) -> str:
        """List all configured devices"""
        if not self.devices:
            return "No devices configured. Add devices using the setup wizard."

        output = "**Smart Home Devices:**\n"
        for device in self.devices.values():
            status = "ON" if device.is_on else "OFF"
            output += f"- {device.name} ({device.device_type}) - {status}\n"
        return output

    def get_status(self, device_name: str = None) -> str:
        """Get device status"""
        if not self.available:
            return self._not_configured_msg()

        if device_name:
            device_id = self._get_device_id(device_name)
            if not device_id:
                return f"Device not found: {device_name}"
            device = self.devices[device_id]
            return self._get_device_status(device)
        else:
            output = "**Device Status:**\n"
            for device in self.devices.values():
                output += self._get_device_status(device) + "\n"
            return output

    def _get_device_status(self, device: TuyaDevice) -> str:
        """Get single device status"""
        try:
            d = self._get_tuya_device(device)
            if d:
                status = d.status()
                device.is_on = status.get('dps', {}).get('1', False)
                return f"{device.name}: {'ON' if device.is_on else 'OFF'}"
        except Exception as e:
            return f"{device.name}: Error - {e}"
        return f"{device.name}: Unknown"

    def _not_configured_msg(self) -> str:
        return """Smart home not configured.

To set up:
1. Install tinytuya: pip install tinytuya
2. Run the setup wizard: python tuya_smart_home.py --setup
3. Or manually add devices to ~/.alexandra/tuya_config.json

Need help? Ask me 'how do I set up smart home'"""

    # ==================== COMMAND PARSER ====================

    def parse_command(self, text: str) -> Optional[str]:
        """Parse natural language commands"""
        text_lower = text.lower().strip()

        # List devices
        if any(p in text_lower for p in ['list devices', 'show devices', 'what devices', 'my devices']):
            return self.list_devices()

        # Turn on
        if 'turn on' in text_lower:
            # Extract device name
            device = self._extract_device_name(text_lower, 'turn on')
            return self.turn_on(device)

        # Turn off
        if 'turn off' in text_lower:
            device = self._extract_device_name(text_lower, 'turn off')
            return self.turn_off(device)

        # Set color
        if 'set' in text_lower and any(c in text_lower for c in ['color', 'colour', 'to red', 'to blue', 'to green', 'to purple', 'to pink', 'to orange', 'to yellow']):
            device, color = self._extract_device_and_color(text_lower)
            if color:
                return self.set_color(device, color)

        # Change color
        if 'change' in text_lower and any(c in text_lower for c in ['color', 'colour']):
            device, color = self._extract_device_and_color(text_lower)
            if color:
                return self.set_color(device, color)

        # Brightness
        if 'dim' in text_lower or 'brightness' in text_lower or 'bright' in text_lower:
            import re
            match = re.search(r'(\d+)\s*%?', text_lower)
            if match:
                brightness = int(match.group(1))
                device = self._extract_device_name(text_lower, 'dim|brightness|bright')
                return self.set_brightness(device, brightness)

        # Make it [color]
        if text_lower.startswith('make') and 'light' in text_lower:
            for color in ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'yellow', 'white', 'cyan']:
                if color in text_lower:
                    device = self._extract_device_name(text_lower, 'make')
                    return self.set_color(device, color)

        return None

    def _extract_device_name(self, text: str, action_pattern: str) -> str:
        """Extract device name from command"""
        import re

        # Remove the action
        text = re.sub(action_pattern, '', text, flags=re.IGNORECASE).strip()

        # Remove common words
        for word in ['the', 'light', 'lights', 'lamp', 'lamps', 'please', 'can you', 'could you']:
            text = text.replace(word, '').strip()

        # Check for room names
        rooms = ['bedroom', 'living room', 'tv room', 'kitchen', 'bathroom', 'office', 'all']
        for room in rooms:
            if room in text:
                return room

        return text.strip() or 'all'

    def _extract_device_and_color(self, text: str) -> tuple:
        """Extract device name and color from command"""
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
                  'cyan', 'white', 'magenta', 'teal', 'coral', 'gold', 'lavender', 'violet']

        found_color = None
        for color in colors:
            if color in text:
                found_color = color
                break

        device = self._extract_device_name(text, 'set|change|make')

        return device, found_color


# ==================== SETUP WIZARD ====================

def run_setup_wizard():
    """Interactive setup wizard for Tuya devices"""
    print("""
╔════════════════════════════════════════════════════════════╗
║          Tuya Smart Home Setup Wizard                      ║
╚════════════════════════════════════════════════════════════╝

This wizard will help you configure your Tuya/Smart Life devices.

You'll need:
1. Your device ID (from Tuya IoT Platform or tinytuya wizard)
2. Your device's local key
3. Your device's IP address (optional, can auto-discover)

If you don't have these, run: python -m tinytuya wizard
""")

    smart_home = TuyaSmartHome()

    while True:
        print("\nOptions:")
        print("1. Add a device manually")
        print("2. Run tinytuya wizard (discovers devices)")
        print("3. List configured devices")
        print("4. Test a device")
        print("5. Save and exit")

        choice = input("\nChoice [1-5]: ").strip()

        if choice == '1':
            print("\n--- Add Device ---")
            device_id = input("Device ID: ").strip()
            name = input("Device name (e.g., 'TV Room Light'): ").strip()
            local_key = input("Local key: ").strip()
            ip = input("IP address (leave blank for auto): ").strip()
            device_type = input("Type (light/switch/plug) [light]: ").strip() or 'light'

            result = smart_home.add_device(device_id, name, local_key, ip, device_type)
            print(result)

        elif choice == '2':
            print("\nRunning tinytuya wizard...")
            print("This will scan your network for Tuya devices.")
            print("You'll need your Tuya IoT Platform credentials.")
            os.system("python -m tinytuya wizard")

        elif choice == '3':
            print(smart_home.list_devices())

        elif choice == '4':
            name = input("Device name to test: ").strip()
            print("Testing turn on...")
            print(smart_home.turn_on(name))
            input("Press Enter to turn off...")
            print(smart_home.turn_off(name))

        elif choice == '5':
            smart_home.save_config()
            print("Configuration saved!")
            break

        else:
            print("Invalid choice")


# ==================== CLI ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup':
            run_setup_wizard()
        elif sys.argv[1] == '--list':
            sh = TuyaSmartHome()
            print(sh.list_devices())
        elif sys.argv[1] == '--on':
            sh = TuyaSmartHome()
            device = sys.argv[2] if len(sys.argv) > 2 else 'all'
            print(sh.turn_on(device))
        elif sys.argv[1] == '--off':
            sh = TuyaSmartHome()
            device = sys.argv[2] if len(sys.argv) > 2 else 'all'
            print(sh.turn_off(device))
        elif sys.argv[1] == '--color':
            sh = TuyaSmartHome()
            color = sys.argv[2] if len(sys.argv) > 2 else 'white'
            device = sys.argv[3] if len(sys.argv) > 3 else 'all'
            print(sh.set_color(device, color))
    else:
        print("""
Tuya Smart Home Controller

Usage:
  python tuya_smart_home.py --setup     Run setup wizard
  python tuya_smart_home.py --list      List devices
  python tuya_smart_home.py --on [device]   Turn on
  python tuya_smart_home.py --off [device]  Turn off
  python tuya_smart_home.py --color <color> [device]  Set color
""")
