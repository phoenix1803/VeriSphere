"""
Geolocation verification tool using Google Maps API
"""
import asyncio
import aiohttp
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from defame.core.interfaces import BaseEvidenceTool
from defame.core.models import Claim, Evidence
from defame.utils.logger import get_logger
from defame.utils.helpers import retry_async, RateLimiter

logger = get_logger(__name__)


class GeolocateTool(BaseEvidenceTool):
    """Geolocation verification tool"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('google_maps_api_key')
        self.base_url = 'https://maps.googleapis.com/maps/api'
        self.timeout = config.get('timeout', 30)
        self.rate_limiter = RateLimiter(
            max_calls=config.get('rate_limit_per_minute', 100),
            time_window=60.0
        )
    
    async def gather_evidence(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """Gather geolocation evidence"""
        try:
            evidence_list = []
            
            # Extract coordinates and location names from query/claim
            locations = self._extract_locations(f"{claim.content} {query}")
            
            for location in locations:
                await self.rate_limiter.acquire()
                
                if location['type'] == 'coordinates':
                    evidence = await self._verify_coordinates(claim, location)
                elif location['type'] == 'place_name':
                    evidence = await self._verify_place_name(claim, location)
                
                if evidence:
                    evidence_list.extend(evidence)
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Geolocation verification failed: {e}")
            return []
    
    def _extract_locations(self, text: str) -> List[Dict[str, Any]]:
        """Extract coordinates and place names from text"""
        locations = []
        
        # Extract coordinates (lat, lng)
        coord_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'
        for match in re.finditer(coord_pattern, text):
            lat, lng = float(match.group(1)), float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                locations.append({
                    'type': 'coordinates',
                    'lat': lat,
                    'lng': lng,
                    'text': match.group(0)
                })
        
        # Extract potential place names (simple heuristic)
        place_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_places = re.findall(place_pattern, text)
        
        # Filter for likely place names (this is simplified)
        place_keywords = ['city', 'town', 'village', 'street', 'avenue', 'road', 'park', 'building']
        for place in potential_places:
            if len(place.split()) <= 3 and any(keyword in text.lower() for keyword in place_keywords):
                locations.append({
                    'type': 'place_name',
                    'name': place,
                    'text': place
                })
        
        return locations
    
    @retry_async(max_attempts=3, delay=1.0)
    async def _verify_coordinates(self, claim: Claim, location: Dict[str, Any]) -> List[Evidence]:
        """Verify coordinates using reverse geocoding"""
        try:
            lat, lng = location['lat'], location['lng']
            
            # Reverse geocoding
            url = f"{self.base_url}/geocode/json"
            params = {
                'latlng': f"{lat},{lng}",
                'key': self.api_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return await self._process_geocoding_result(claim, data, location)
            
            return []
            
        except Exception as e:
            logger.warning(f"Coordinate verification failed: {e}")
            return []
    
    @retry_async(max_attempts=3, delay=1.0)
    async def _verify_place_name(self, claim: Claim, location: Dict[str, Any]) -> List[Evidence]:
        """Verify place name using geocoding"""
        try:
            place_name = location['name']
            
            # Geocoding
            url = f"{self.base_url}/geocode/json"
            params = {
                'address': place_name,
                'key': self.api_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return await self._process_geocoding_result(claim, data, location)
            
            return []
            
        except Exception as e:
            logger.warning(f"Place name verification failed: {e}")
            return []
    
    async def _process_geocoding_result(self, claim: Claim, data: Dict[str, Any], location: Dict[str, Any]) -> List[Evidence]:
        """Process geocoding API results"""
        evidence_list = []
        
        try:
            results = data.get('results', [])
            if not results:
                return []
            
            for result in results[:3]:  # Top 3 results
                formatted_address = result.get('formatted_address', '')
                geometry = result.get('geometry', {})
                location_data = geometry.get('location', {})
                
                # Calculate credibility based on location type and accuracy
                location_type = geometry.get('location_type', 'UNKNOWN')
                credibility_score = {
                    'ROOFTOP': 0.95,
                    'RANGE_INTERPOLATED': 0.85,
                    'GEOMETRIC_CENTER': 0.75,
                    'APPROXIMATE': 0.65
                }.get(location_type, 0.5)
                
                evidence = Evidence(
                    source='Google Maps Geocoding',
                    content=f"Location verified: {formatted_address}",
                    credibility_score=credibility_score,
                    relevance_score=0.9,
                    evidence_type='geolocation',
                    metadata={
                        'original_location': location,
                        'formatted_address': formatted_address,
                        'coordinates': {
                            'lat': location_data.get('lat'),
                            'lng': location_data.get('lng')
                        },
                        'location_type': location_type,
                        'place_types': result.get('types', [])
                    }
                )
                evidence_list.append(evidence)
            
        except Exception as e:
            logger.warning(f"Failed to process geocoding result: {e}")
        
        return evidence_list
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information"""
        return {
            'tool_id': self.tool_id,
            'name': 'GeolocateTool',
            'description': 'Geolocation verification using Google Maps API',
            'capabilities': [
                'coordinate_verification',
                'place_name_verification',
                'reverse_geocoding',
                'location_accuracy_assessment'
            ],
            'rate_limit': '100 requests per minute'
        }
    
    async def health_check(self) -> bool:
        """Check if Maps API is available"""
        try:
            # Test with a known location
            url = f"{self.base_url}/geocode/json"
            params = {
                'address': 'New York',
                'key': self.api_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('status') == 'OK'
            
            return False
            
        except Exception as e:
            logger.error(f"Geolocation tool health check failed: {e}")
            self.is_available = False
            return False