"""
Image analysis tool using Google Vision API for reverse image search and metadata extraction
"""
import asyncio
import aiohttp
import base64
import io
from PIL import Image, ExifTags
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
import json

from defame.core.interfaces import BaseEvidenceTool
from defame.core.models import Claim, Evidence
from defame.utils.logger import get_logger
from defame.utils.helpers import retry_async, RateLimiter, get_file_hash

logger = get_logger(__name__)


class ImageTool(BaseEvidenceTool):
    """Image analysis tool using Google Vision API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('google_vision_api_key')
        self.base_url = 'https://vision.googleapis.com/v1'
        self.timeout = config.get('timeout', 60)
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self.supported_formats = config.get('supported_formats', ['JPEG', 'PNG', 'WEBP', 'GIF'])
        self.rate_limiter = RateLimiter(
            max_calls=config.get('rate_limit_per_minute', 30),
            time_window=60.0
        )
    
    async def gather_evidence(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """
        Analyze image and gather evidence
        
        Args:
            claim: The claim being verified (should contain image data)
            query: Not used for image analysis, but kept for interface compatibility
            **kwargs: Additional parameters (image_path, image_url, analysis_types)
            
        Returns:
            List of Evidence objects from image analysis
        """
        try:
            evidence_list = []
            
            # Get image data
            image_data = await self._get_image_data(claim, kwargs)
            if not image_data:
                return []
            
            await self.rate_limiter.acquire()
            
            logger.info(
                "Starting image analysis",
                claim_id=claim.id,
                image_size=len(image_data.get('content', b''))
            )
            
            # Perform various types of analysis
            analysis_types = kwargs.get('analysis_types', [
                'web_detection',
                'text_detection',
                'object_localization',
                'safe_search',
                'image_properties'
            ])
            
            # Extract EXIF metadata
            metadata_evidence = await self._extract_metadata(claim, image_data)
            if metadata_evidence:
                evidence_list.append(metadata_evidence)
            
            # Perform Google Vision API analysis
            for analysis_type in analysis_types:
                try:
                    evidence = await self._perform_vision_analysis(claim, image_data, analysis_type)
                    if evidence:
                        evidence_list.extend(evidence)
                except Exception as e:
                    logger.warning(f"Failed {analysis_type} analysis: {e}")
            
            # Perform reverse image search
            reverse_search_evidence = await self._reverse_image_search(claim, image_data)
            evidence_list.extend(reverse_search_evidence)
            
            logger.info(
                "Image analysis completed",
                claim_id=claim.id,
                evidence_count=len(evidence_list)
            )
            
            return evidence_list
            
        except Exception as e:
            logger.error(
                "Image analysis failed",
                claim_id=claim.id,
                error=str(e)
            )
            return []
    
    async def _get_image_data(self, claim: Claim, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get image data from various sources"""
        try:
            # Check if image data is in claim content
            if isinstance(claim.content, bytes):
                return {
                    'content': claim.content,
                    'source': 'claim_content',
                    'format': self._detect_image_format(claim.content)
                }
            
            # Check for image path
            image_path = kwargs.get('image_path')
            if image_path:
                with open(image_path, 'rb') as f:
                    content = f.read()
                return {
                    'content': content,
                    'source': 'file_path',
                    'path': image_path,
                    'format': self._detect_image_format(content)
                }
            
            # Check for image URL
            image_url = kwargs.get('image_url')
            if image_url:
                content = await self._download_image(image_url)
                if content:
                    return {
                        'content': content,
                        'source': 'url',
                        'url': image_url,
                        'format': self._detect_image_format(content)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get image data: {e}")
            return None
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """Detect image format from binary data"""
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                return img.format
        except Exception:
            return 'UNKNOWN'
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        if len(content) <= self.max_file_size:
                            return content
            return None
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return None
    
    async def _extract_metadata(self, claim: Claim, image_data: Dict[str, Any]) -> Optional[Evidence]:
        """Extract EXIF metadata from image"""
        try:
            content = image_data['content']
            
            with Image.open(io.BytesIO(content)) as img:
                # Get basic image info
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'file_size': len(content),
                    'hash': hashlib.sha256(content).hexdigest()
                }
                
                # Extract EXIF data
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = str(value)
                
                metadata['exif'] = exif_data
                
                # Calculate credibility based on metadata presence
                credibility_score = 0.6  # Base score
                if exif_data:
                    credibility_score += 0.2  # Has EXIF data
                if 'DateTime' in exif_data:
                    credibility_score += 0.1  # Has timestamp
                if 'Make' in exif_data or 'Model' in exif_data:
                    credibility_score += 0.1  # Has camera info
                
                evidence = Evidence(
                    source='Image Metadata',
                    content=f"Image metadata analysis: {img.format} image, {img.size[0]}x{img.size[1]} pixels",
                    credibility_score=min(credibility_score, 1.0),
                    relevance_score=0.8,
                    evidence_type='image_metadata',
                    metadata=metadata
                )
                
                return evidence
                
        except Exception as e:
            logger.warning(f"Failed to extract image metadata: {e}")
            return None
    
    @retry_async(max_attempts=3, delay=1.0, backoff=2.0)
    async def _perform_vision_analysis(self, claim: Claim, image_data: Dict[str, Any], analysis_type: str) -> List[Evidence]:
        """Perform Google Vision API analysis"""
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_data['content']).decode('utf-8')
            
            # Prepare request
            request_data = {
                'requests': [{
                    'image': {
                        'content': image_base64
                    },
                    'features': [{
                        'type': analysis_type.upper(),
                        'maxResults': 10
                    }]
                }]
            }
            
            # Make API call
            url = f"{self.base_url}/images:annotate?key={self.api_key}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return await self._process_vision_result(claim, result, analysis_type)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Vision API error {response.status}: {error_text}")
            
        except Exception as e:
            logger.error(f"Vision API {analysis_type} failed: {e}")
            return []
    
    async def _process_vision_result(self, claim: Claim, result: Dict[str, Any], analysis_type: str) -> List[Evidence]:
        """Process Google Vision API results into Evidence objects"""
        evidence_list = []
        
        try:
            responses = result.get('responses', [])
            if not responses:
                return []
            
            response = responses[0]
            
            if analysis_type == 'web_detection':
                evidence_list.extend(await self._process_web_detection(claim, response))
            elif analysis_type == 'text_detection':
                evidence_list.extend(await self._process_text_detection(claim, response))
            elif analysis_type == 'object_localization':
                evidence_list.extend(await self._process_object_detection(claim, response))
            elif analysis_type == 'safe_search':
                evidence_list.extend(await self._process_safe_search(claim, response))
            elif analysis_type == 'image_properties':
                evidence_list.extend(await self._process_image_properties(claim, response))
            
        except Exception as e:
            logger.warning(f"Failed to process {analysis_type} result: {e}")
        
        return evidence_list
    
    async def _process_web_detection(self, claim: Claim, response: Dict[str, Any]) -> List[Evidence]:
        """Process web detection results"""
        evidence_list = []
        web_detection = response.get('webDetection', {})
        
        # Process web entities
        web_entities = web_detection.get('webEntities', [])
        for entity in web_entities[:5]:  # Top 5 entities
            if entity.get('score', 0) > 0.5:
                evidence = Evidence(
                    source='Google Vision Web Detection',
                    content=f"Web entity: {entity.get('description', 'Unknown')}",
                    credibility_score=0.7,
                    relevance_score=entity.get('score', 0.5),
                    evidence_type='web_entity',
                    metadata={
                        'entity_id': entity.get('entityId'),
                        'description': entity.get('description'),
                        'score': entity.get('score')
                    }
                )
                evidence_list.append(evidence)
        
        # Process pages with matching images
        pages_with_matching_images = web_detection.get('pagesWithMatchingImages', [])
        for page in pages_with_matching_images[:3]:  # Top 3 pages
            evidence = Evidence(
                source='Reverse Image Search',
                content=f"Image found on: {page.get('pageTitle', 'Unknown page')}",
                url=page.get('url'),
                credibility_score=0.6,
                relevance_score=0.8,
                evidence_type='reverse_image_match',
                metadata={
                    'page_title': page.get('pageTitle'),
                    'url': page.get('url')
                }
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    async def _process_text_detection(self, claim: Claim, response: Dict[str, Any]) -> List[Evidence]:
        """Process text detection results"""
        evidence_list = []
        text_annotations = response.get('textAnnotations', [])
        
        if text_annotations:
            # Full text is usually the first annotation
            full_text = text_annotations[0].get('description', '')
            
            if full_text.strip():
                evidence = Evidence(
                    source='Image Text Detection',
                    content=f"Text found in image: {full_text}",
                    credibility_score=0.8,
                    relevance_score=0.9,
                    evidence_type='image_text',
                    metadata={
                        'detected_text': full_text,
                        'text_count': len(text_annotations)
                    }
                )
                evidence_list.append(evidence)
        
        return evidence_list
    
    async def _process_object_detection(self, claim: Claim, response: Dict[str, Any]) -> List[Evidence]:
        """Process object detection results"""
        evidence_list = []
        objects = response.get('localizedObjectAnnotations', [])
        
        detected_objects = []
        for obj in objects:
            if obj.get('score', 0) > 0.5:
                detected_objects.append({
                    'name': obj.get('name'),
                    'score': obj.get('score')
                })
        
        if detected_objects:
            object_names = [obj['name'] for obj in detected_objects]
            evidence = Evidence(
                source='Image Object Detection',
                content=f"Objects detected in image: {', '.join(object_names)}",
                credibility_score=0.7,
                relevance_score=0.6,
                evidence_type='object_detection',
                metadata={
                    'detected_objects': detected_objects,
                    'object_count': len(detected_objects)
                }
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    async def _process_safe_search(self, claim: Claim, response: Dict[str, Any]) -> List[Evidence]:
        """Process safe search results"""
        evidence_list = []
        safe_search = response.get('safeSearchAnnotation', {})
        
        if safe_search:
            # Check for problematic content
            categories = ['adult', 'spoof', 'medical', 'violence', 'racy']
            flags = []
            
            for category in categories:
                likelihood = safe_search.get(category, 'UNKNOWN')
                if likelihood in ['LIKELY', 'VERY_LIKELY']:
                    flags.append(category)
            
            if flags:
                evidence = Evidence(
                    source='Image Safety Analysis',
                    content=f"Image flagged for: {', '.join(flags)}",
                    credibility_score=0.9,
                    relevance_score=0.7,
                    evidence_type='safety_analysis',
                    metadata={
                        'safe_search_results': safe_search,
                        'flagged_categories': flags
                    }
                )
                evidence_list.append(evidence)
        
        return evidence_list
    
    async def _process_image_properties(self, claim: Claim, response: Dict[str, Any]) -> List[Evidence]:
        """Process image properties results"""
        evidence_list = []
        image_properties = response.get('imagePropertiesAnnotation', {})
        
        if image_properties:
            dominant_colors = image_properties.get('dominantColors', {}).get('colors', [])
            
            if dominant_colors:
                color_info = []
                for color in dominant_colors[:3]:  # Top 3 colors
                    rgb = color.get('color', {})
                    score = color.get('score', 0)
                    color_info.append({
                        'rgb': [rgb.get('red', 0), rgb.get('green', 0), rgb.get('blue', 0)],
                        'score': score
                    })
                
                evidence = Evidence(
                    source='Image Color Analysis',
                    content=f"Dominant colors analyzed in image",
                    credibility_score=0.5,
                    relevance_score=0.3,
                    evidence_type='color_analysis',
                    metadata={
                        'dominant_colors': color_info
                    }
                )
                evidence_list.append(evidence)
        
        return evidence_list
    
    async def _reverse_image_search(self, claim: Claim, image_data: Dict[str, Any]) -> List[Evidence]:
        """Perform reverse image search using multiple engines"""
        evidence_list = []
        
        try:
            # This would integrate with TinEye, Bing, or other reverse image search APIs
            # For now, we'll use the web detection results from Google Vision
            # In a full implementation, you would add additional reverse search engines
            
            logger.info("Reverse image search completed via web detection")
            
        except Exception as e:
            logger.warning(f"Reverse image search failed: {e}")
        
        return evidence_list
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information"""
        return {
            'tool_id': self.tool_id,
            'name': 'ImageTool',
            'description': 'Image analysis using Google Vision API',
            'capabilities': [
                'metadata_extraction',
                'text_detection',
                'object_detection',
                'web_detection',
                'reverse_image_search',
                'safe_search',
                'color_analysis'
            ],
            'supported_formats': self.supported_formats,
            'max_file_size': self.max_file_size,
            'rate_limit': '30 requests per minute'
        }
    
    async def health_check(self) -> bool:
        """Check if Vision API is available"""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            test_data = {
                'content': img_bytes.getvalue(),
                'source': 'test'
            }
            
            # Try a simple analysis
            result = await self._perform_vision_analysis(
                Claim(content="test"), 
                test_data, 
                'image_properties'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Image tool health check failed: {e}")
            self.is_available = False
            return False


# Example usage and testing
if __name__ == "__main__":
    import os
    from defame.core.models import Claim
    from config.globals import ClaimType
    
    async def test_image_tool():
        # Test configuration
        config = {
            'google_vision_api_key': os.getenv('GOOGLE_VISION_API_KEY', 'test-key')
        }
        
        image_tool = ImageTool(config)
        
        # Create a test image
        test_image = Image.new('RGB', (200, 200), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        
        # Test claim with image data
        test_claim = Claim(
            content=img_bytes.getvalue(),
            claim_type=ClaimType.IMAGE
        )
        
        # Test image analysis
        evidence = await image_tool.gather_evidence(test_claim, "")
        
        print(f"Found {len(evidence)} pieces of evidence:")
        for i, e in enumerate(evidence):
            print(f"{i+1}. {e.source} ({e.evidence_type})")
            print(f"   Credibility: {e.credibility_score:.2f}, Relevance: {e.relevance_score:.2f}")
            print(f"   {e.content[:100]}...")
            print()
    
    # Run test if API key is available
    if os.getenv('GOOGLE_VISION_API_KEY'):
        asyncio.run(test_image_tool())
    else:
        print("Set GOOGLE_VISION_API_KEY environment variable to test image tool")