#!/usr/bin/env python3
"""
JARVIS Navigation & Traffic System
- Directions between locations
- Traffic conditions
- Travel time estimates
- Map integration
"""

import os
import re
import json
import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import requests

logger = logging.getLogger("JarvisNavigation")

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenRouteService API (free, get key at https://openrouteservice.org/)
ORS_API_KEY = os.environ.get("ORS_API_KEY", "")

# Google Maps API (optional, for enhanced features)
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Default home location
DEFAULT_HOME = "Ridgway, PA"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RouteStep:
    """Single step in a route"""
    instruction: str
    distance: float  # meters
    duration: float  # seconds
    road_name: str


@dataclass
class RouteInfo:
    """Complete route information"""
    origin: str
    destination: str
    distance: float  # kilometers
    duration: float  # minutes
    duration_traffic: Optional[float]  # with traffic
    steps: List[RouteStep]
    summary: str
    map_url: str
    traffic_status: str  # light, moderate, heavy
    departure_time: Optional[datetime]
    arrival_time: Optional[datetime]
    map_html: str = ""  # Embeddable map HTML


# ============================================================================
# GEOCODING
# ============================================================================

class Geocoder:
    """Convert addresses to coordinates"""

    def __init__(self):
        self.cache = {}

    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Get lat/lon for an address"""
        if address in self.cache:
            return self.cache[address]

        # Try Nominatim (free, OpenStreetMap)
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": address,
                "format": "json",
                "limit": 1
            }
            headers = {"User-Agent": "JarvisAI/1.0"}

            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    self.cache[address] = (lat, lon)
                    return (lat, lon)
        except Exception as e:
            logger.error(f"Geocoding error: {e}")

        return None

    def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """Get address from coordinates"""
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": lat,
                "lon": lon,
                "format": "json"
            }
            headers = {"User-Agent": "JarvisAI/1.0"}

            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("display_name", "")
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")

        return None


# ============================================================================
# DIRECTIONS SERVICE
# ============================================================================

class DirectionsService:
    """Get directions between locations"""

    def __init__(self):
        self.geocoder = Geocoder()

    def get_directions(self, origin: str, destination: str,
                       mode: str = "driving") -> Optional[RouteInfo]:
        """
        Get directions from origin to destination.

        Args:
            origin: Starting location (address or place name)
            destination: End location
            mode: driving, walking, cycling

        Returns:
            RouteInfo with directions, or None if failed
        """
        # Geocode addresses
        origin_coords = self.geocoder.geocode(origin)
        dest_coords = self.geocoder.geocode(destination)

        if not origin_coords or not dest_coords:
            logger.error(f"Could not geocode: origin={origin_coords}, dest={dest_coords}")
            return None

        # Try OpenRouteService first
        if ORS_API_KEY:
            route = self._get_ors_directions(origin_coords, dest_coords, mode)
            if route:
                route.origin = origin
                route.destination = destination
                route.map_url = self._get_google_maps_url(origin, destination, mode)
                route.map_html = self.get_embed_map_html(origin, destination)
                return route

        # Fallback to OSRM (free, no API key)
        route = self._get_osrm_directions(origin_coords, dest_coords, mode)
        if route:
            route.origin = origin
            route.destination = destination
            route.map_url = self._get_google_maps_url(origin, destination, mode)
            route.map_html = self.get_embed_map_html(origin, destination)
            return route

        return None

    def _get_osrm_directions(self, origin: Tuple[float, float],
                              destination: Tuple[float, float],
                              mode: str) -> Optional[RouteInfo]:
        """Get directions from OSRM (Open Source Routing Machine)"""
        try:
            # OSRM profile mapping
            profile_map = {
                "driving": "car",
                "walking": "foot",
                "cycling": "bike"
            }
            profile = profile_map.get(mode, "car")

            # OSRM demo server (for production, use your own)
            url = f"https://router.project-osrm.org/route/v1/{profile}/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
            params = {
                "overview": "full",
                "steps": "true",
                "annotations": "true"
            }

            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            if data.get("code") != "Ok":
                return None

            route = data["routes"][0]
            legs = route["legs"][0]

            # Parse steps
            steps = []
            for step in legs.get("steps", []):
                instruction = step.get("maneuver", {}).get("instruction", "")
                if not instruction:
                    # Build instruction from type
                    maneuver = step.get("maneuver", {})
                    mtype = maneuver.get("type", "")
                    modifier = maneuver.get("modifier", "")
                    road = step.get("name", "the road")

                    if mtype == "turn":
                        instruction = f"Turn {modifier} onto {road}"
                    elif mtype == "depart":
                        instruction = f"Head towards {road}"
                    elif mtype == "arrive":
                        instruction = "Arrive at destination"
                    elif mtype == "merge":
                        instruction = f"Merge onto {road}"
                    elif mtype == "fork":
                        instruction = f"Take the {modifier} fork onto {road}"
                    else:
                        instruction = f"Continue on {road}"

                steps.append(RouteStep(
                    instruction=instruction,
                    distance=step.get("distance", 0),
                    duration=step.get("duration", 0),
                    road_name=step.get("name", "")
                ))

            # Calculate times
            duration_min = route["duration"] / 60
            distance_km = route["distance"] / 1000

            # Estimate traffic (simple heuristic based on time of day)
            now = datetime.now()
            hour = now.hour
            is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 18)
            is_weekend = now.weekday() >= 5

            if is_weekend:
                traffic_multiplier = 1.0
                traffic_status = "light"
            elif is_rush_hour:
                traffic_multiplier = 1.4
                traffic_status = "heavy"
            else:
                traffic_multiplier = 1.1
                traffic_status = "moderate"

            duration_traffic = duration_min * traffic_multiplier

            return RouteInfo(
                origin="",
                destination="",
                distance=distance_km,
                duration=duration_min,
                duration_traffic=duration_traffic,
                steps=steps,
                summary=f"{distance_km:.1f} km, about {int(duration_traffic)} minutes",
                map_url="",
                traffic_status=traffic_status,
                departure_time=now,
                arrival_time=now + timedelta(minutes=duration_traffic)
            )

        except Exception as e:
            logger.error(f"OSRM directions error: {e}")
            return None

    def _get_ors_directions(self, origin: Tuple[float, float],
                            destination: Tuple[float, float],
                            mode: str) -> Optional[RouteInfo]:
        """Get directions from OpenRouteService"""
        try:
            profile_map = {
                "driving": "driving-car",
                "walking": "foot-walking",
                "cycling": "cycling-regular"
            }
            profile = profile_map.get(mode, "driving-car")

            url = f"https://api.openrouteservice.org/v2/directions/{profile}"
            headers = {
                "Authorization": ORS_API_KEY,
                "Content-Type": "application/json"
            }
            body = {
                "coordinates": [
                    [origin[1], origin[0]],
                    [destination[1], destination[0]]
                ],
                "instructions": True,
                "units": "km"
            }

            resp = requests.post(url, headers=headers, json=body, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            route = data["routes"][0]
            summary = route["summary"]

            # Parse steps
            steps = []
            for segment in route.get("segments", []):
                for step in segment.get("steps", []):
                    steps.append(RouteStep(
                        instruction=step.get("instruction", ""),
                        distance=step.get("distance", 0) * 1000,  # km to m
                        duration=step.get("duration", 0),
                        road_name=step.get("name", "")
                    ))

            duration_min = summary["duration"] / 60
            distance_km = summary["distance"]

            return RouteInfo(
                origin="",
                destination="",
                distance=distance_km,
                duration=duration_min,
                duration_traffic=duration_min * 1.1,  # Estimate
                steps=steps,
                summary=f"{distance_km:.1f} km, about {int(duration_min)} minutes",
                map_url="",
                traffic_status="moderate",
                departure_time=datetime.now(),
                arrival_time=datetime.now() + timedelta(minutes=duration_min)
            )

        except Exception as e:
            logger.error(f"ORS directions error: {e}")
            return None

    def _get_google_maps_url(self, origin: str, destination: str, mode: str) -> str:
        """Generate Google Maps URL for directions"""
        mode_map = {
            "driving": "driving",
            "walking": "walking",
            "cycling": "bicycling",
            "transit": "transit"
        }
        gmode = mode_map.get(mode, "driving")

        origin_encoded = urllib.parse.quote(origin)
        dest_encoded = urllib.parse.quote(destination)

        return f"https://www.google.com/maps/dir/?api=1&origin={origin_encoded}&destination={dest_encoded}&travelmode={gmode}"

    def get_static_map_url(self, origin_coords: Tuple[float, float],
                           dest_coords: Tuple[float, float],
                           width: int = 600, height: int = 400) -> str:
        """Generate static map image URL showing route"""
        # Use OpenStreetMap static map with markers
        # Calculate center point
        center_lat = (origin_coords[0] + dest_coords[0]) / 2
        center_lon = (origin_coords[1] + dest_coords[1]) / 2

        # Calculate zoom level based on distance
        import math
        lat_diff = abs(origin_coords[0] - dest_coords[0])
        lon_diff = abs(origin_coords[1] - dest_coords[1])
        max_diff = max(lat_diff, lon_diff)

        if max_diff > 5:
            zoom = 6
        elif max_diff > 2:
            zoom = 7
        elif max_diff > 1:
            zoom = 8
        elif max_diff > 0.5:
            zoom = 9
        elif max_diff > 0.2:
            zoom = 10
        else:
            zoom = 11

        # OpenStreetMap static map URL with markers
        # Using staticmapmaker or similar service
        map_url = (
            f"https://staticmap.openstreetmap.de/staticmap.php?"
            f"center={center_lat},{center_lon}&zoom={zoom}&size={width}x{height}&maptype=mapnik"
            f"&markers={origin_coords[0]},{origin_coords[1]},lightblue"
            f"&markers={dest_coords[0]},{dest_coords[1]},red"
        )

        return map_url

    def get_embed_map_html(self, origin: str, destination: str) -> str:
        """Generate embeddable map HTML using Google Maps"""
        origin_encoded = urllib.parse.quote(origin)
        dest_encoded = urllib.parse.quote(destination)

        # Use Google Maps embed - shows actual directions with route
        google_maps_url = f"https://www.google.com/maps/embed/v1/directions?key={GOOGLE_MAPS_API_KEY}&origin={origin_encoded}&destination={dest_encoded}&mode=driving"

        # Fallback to basic Google Maps if embed doesn't work
        basic_maps_url = f"https://maps.google.com/maps?q={dest_encoded}&t=m&z=12&output=embed"

        html = f'''
        <div style="width:100%; height:400px; border-radius:10px; overflow:hidden; border:2px solid #0f3460; background:#1a1a2e;">
            <iframe
                width="100%"
                height="100%"
                frameborder="0"
                style="border:0; border-radius:10px;"
                loading="lazy"
                allowfullscreen
                referrerpolicy="no-referrer-when-downgrade"
                src="{basic_maps_url}">
            </iframe>
        </div>
        <div style="text-align:center; margin-top:10px;">
            <a href="https://www.google.com/maps/dir/?api=1&origin={origin_encoded}&destination={dest_encoded}&travelmode=driving"
               target="_blank"
               style="background: linear-gradient(135deg, #4285f4, #34a853); color:white; padding:10px 20px;
                      border-radius:25px; text-decoration:none; font-weight:bold; display:inline-block;">
               🗺️ Open Full Directions in Google Maps
            </a>
        </div>
        '''
        return html

    def _get_bbox(self, origin: str, destination: str) -> str:
        """Get bounding box for map embed"""
        origin_coords = self.geocoder.geocode(origin)
        dest_coords = self.geocoder.geocode(destination)

        if not origin_coords or not dest_coords:
            return "-80,40,-75,42"  # Default PA area

        min_lon = min(origin_coords[1], dest_coords[1]) - 0.5
        max_lon = max(origin_coords[1], dest_coords[1]) + 0.5
        min_lat = min(origin_coords[0], dest_coords[0]) - 0.3
        max_lat = max(origin_coords[0], dest_coords[0]) + 0.3

        return f"{min_lon},{min_lat},{max_lon},{max_lat}"

    def _get_marker_coords(self, location: str) -> str:
        """Get marker coordinates string"""
        coords = self.geocoder.geocode(location)
        if coords:
            return f"{coords[0]},{coords[1]}"
        return "40.5,-78.5"  # Default


# ============================================================================
# TRAFFIC SERVICE
# ============================================================================

class TrafficService:
    """Get traffic conditions"""

    def __init__(self):
        self.directions = DirectionsService()

    def get_traffic(self, origin: str, destination: str) -> Dict[str, Any]:
        """
        Get traffic conditions between two points.

        Returns dict with:
            - status: light/moderate/heavy
            - delay_minutes: estimated delay
            - incidents: list of traffic incidents
            - recommendation: best time to travel
        """
        route = self.directions.get_directions(origin, destination)

        if not route:
            return {
                "status": "unknown",
                "delay_minutes": 0,
                "message": f"Could not get traffic between {origin} and {destination}"
            }

        # Calculate delay
        normal_duration = route.duration
        with_traffic = route.duration_traffic or route.duration
        delay = max(0, with_traffic - normal_duration)

        # Determine status
        delay_pct = (delay / normal_duration * 100) if normal_duration > 0 else 0

        if delay_pct < 10:
            status = "light"
            status_emoji = "🟢"
        elif delay_pct < 30:
            status = "moderate"
            status_emoji = "🟡"
        else:
            status = "heavy"
            status_emoji = "🔴"

        # Generate recommendation
        now = datetime.now()
        if status == "heavy":
            # Suggest leaving earlier or later
            if now.hour < 12:
                recommendation = "Consider leaving after 10 AM to avoid traffic"
            else:
                recommendation = "Consider waiting until after 7 PM"
        else:
            recommendation = "Traffic looks good for traveling now"

        return {
            "status": status,
            "status_emoji": status_emoji,
            "delay_minutes": int(delay),
            "normal_duration": int(normal_duration),
            "with_traffic_duration": int(with_traffic),
            "recommendation": recommendation,
            "route": route
        }

    def get_commute_status(self, home: str = None, work: str = None) -> str:
        """Get status for regular commute"""
        home = home or DEFAULT_HOME

        if not work:
            return "Work location not configured"

        traffic = self.get_traffic(home, work)

        if "route" not in traffic:
            return traffic.get("message", "Could not check traffic")

        return (
            f"{traffic['status_emoji']} Traffic is {traffic['status']}\n"
            f"Normal: {traffic['normal_duration']} min\n"
            f"Current: {traffic['with_traffic_duration']} min\n"
            f"Delay: +{traffic['delay_minutes']} min\n"
            f"{traffic['recommendation']}"
        )


# ============================================================================
# NAVIGATION ASSISTANT
# ============================================================================

class NavigationAssistant:
    """
    High-level navigation assistant for voice commands.
    """

    def __init__(self):
        self.directions = DirectionsService()
        self.traffic = TrafficService()
        self.home = DEFAULT_HOME

    def set_home(self, address: str):
        """Set home location"""
        self.home = address

    def parse_navigation_request(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language navigation request.

        Examples:
            - "How do I get to Pittsburgh?"
            - "Directions to the airport"
            - "Navigate to 123 Main St"
            - "What's the traffic like to work?"
            - "How long to get to the mall?"
        """
        text_lower = text.lower()

        # Extract destination
        destination = None

        # Pattern: "to [destination]"
        to_match = re.search(r'(?:to|for)\s+(.+?)(?:\?|$|from)', text_lower)
        if to_match:
            destination = to_match.group(1).strip()

        # Pattern: "directions [destination]"
        dir_match = re.search(r'directions?\s+(?:to\s+)?(.+?)(?:\?|$)', text_lower)
        if dir_match and not destination:
            destination = dir_match.group(1).strip()

        # Extract origin (default to home)
        origin = self.home

        from_match = re.search(r'from\s+(.+?)(?:\s+to|\?|$)', text_lower)
        if from_match:
            origin = from_match.group(1).strip()

        # Detect mode
        mode = "driving"
        if any(w in text_lower for w in ["walk", "walking", "on foot"]):
            mode = "walking"
        elif any(w in text_lower for w in ["bike", "bicycle", "cycling"]):
            mode = "cycling"

        # Detect if traffic-only request
        is_traffic_request = any(w in text_lower for w in ["traffic", "congestion", "busy"])

        return {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "is_traffic_request": is_traffic_request
        }

    def get_directions_response(self, text: str) -> Tuple[str, Optional[RouteInfo]]:
        """
        Get directions from natural language request.

        Returns:
            (response_text, route_info)
        """
        parsed = self.parse_navigation_request(text)

        if not parsed["destination"]:
            return "Where would you like directions to?", None

        # Traffic-only request
        if parsed["is_traffic_request"]:
            traffic = self.traffic.get_traffic(parsed["origin"], parsed["destination"])
            if "route" in traffic:
                route = traffic["route"]
                response = (
                    f"{traffic['status_emoji']} Traffic from {parsed['origin']} to {parsed['destination']}:\n\n"
                    f"Status: {traffic['status'].upper()}\n"
                    f"Normal time: {traffic['normal_duration']} minutes\n"
                    f"With traffic: {traffic['with_traffic_duration']} minutes\n"
                    f"Delay: +{traffic['delay_minutes']} minutes\n\n"
                    f"{traffic['recommendation']}\n\n"
                    f"📍 Open in Maps: {route.map_url}"
                )
                return response, route
            else:
                return traffic.get("message", "Could not get traffic info"), None

        # Full directions request
        route = self.directions.get_directions(
            parsed["origin"],
            parsed["destination"],
            parsed["mode"]
        )

        if not route:
            return f"Sorry, I couldn't find directions to {parsed['destination']}", None

        # Build response
        mode_text = {
            "driving": "🚗 Driving",
            "walking": "🚶 Walking",
            "cycling": "🚴 Cycling"
        }.get(parsed["mode"], "🚗 Driving")

        arrival = route.arrival_time.strftime("%I:%M %p") if route.arrival_time else "Unknown"

        response = (
            f"{mode_text} directions to {route.destination}:\n\n"
            f"📍 From: {route.origin}\n"
            f"🎯 To: {route.destination}\n\n"
            f"📏 Distance: {route.distance:.1f} km ({route.distance * 0.621:.1f} miles)\n"
            f"⏱️ Time: {int(route.duration_traffic or route.duration)} minutes\n"
            f"🕐 Arrive by: {arrival}\n"
            f"🚦 Traffic: {route.traffic_status}\n\n"
        )

        # Add first few steps
        if route.steps:
            response += "📝 Route:\n"
            for i, step in enumerate(route.steps[:5], 1):
                dist_mi = step.distance / 1609.34
                response += f"  {i}. {step.instruction} ({dist_mi:.1f} mi)\n"

            if len(route.steps) > 5:
                response += f"  ... and {len(route.steps) - 5} more steps\n"

        response += f"\n📍 Open in Maps: {route.map_url}"

        return response, route

    def get_eta(self, destination: str) -> str:
        """Quick ETA check"""
        route = self.directions.get_directions(self.home, destination)

        if not route:
            return f"Couldn't calculate ETA to {destination}"

        duration = int(route.duration_traffic or route.duration)
        arrival = route.arrival_time.strftime("%I:%M %p") if route.arrival_time else ""

        return f"About {duration} minutes to {destination}. Arrive around {arrival}."


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

navigator = NavigationAssistant()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_directions(origin: str, destination: str, mode: str = "driving") -> Optional[RouteInfo]:
    """Get directions between two locations"""
    return navigator.directions.get_directions(origin, destination, mode)


def get_traffic(origin: str, destination: str) -> Dict[str, Any]:
    """Get traffic conditions"""
    return navigator.traffic.get_traffic(origin, destination)


def parse_and_respond(text: str) -> Tuple[str, Optional[RouteInfo]]:
    """Parse navigation request and get response"""
    return navigator.get_directions_response(text)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Navigation System...")

    # Test geocoding
    print("\n=== Geocoding ===")
    geocoder = Geocoder()
    coords = geocoder.geocode("Pittsburgh, PA")
    print(f"Pittsburgh, PA: {coords}")

    # Test directions
    print("\n=== Directions ===")
    response, route = parse_and_respond("How do I get to Pittsburgh from Ridgway PA?")
    print(response)

    # Test traffic
    print("\n=== Traffic ===")
    response, route = parse_and_respond("What's the traffic like to Erie PA?")
    print(response)

    # Test ETA
    print("\n=== ETA ===")
    print(navigator.get_eta("State College, PA"))
