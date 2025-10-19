"""
PII (Personally Identifiable Information) detection and redaction
"""
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json

from defame.utils.logger import get_logger

logger = get_logger(__name__)


class PIIType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    URL = "url"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """Detected PII match"""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float
    context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "type": self.pii_type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "context": self.context
        }


class PIIDetector:
    """Advanced PII detection and redaction system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled_types = set(self.config.get('enabled_types', [t.value for t in PIIType]))
        self.redaction_char = self.config.get('redaction_char', '*')
        self.preserve_format = self.config.get('preserve_format', True)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Common names for enhanced detection (simplified list)
        self.common_names = {
            'first_names': {
                'john', 'jane', 'michael', 'sarah', 'david', 'mary', 'james', 'patricia',
                'robert', 'jennifer', 'william', 'linda', 'richard', 'elizabeth', 'joseph',
                'barbara', 'thomas', 'susan', 'charles', 'jessica', 'christopher', 'karen'
            },
            'last_names': {
                'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller',
                'davis', 'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez',
                'wilson', 'anderson', 'thomas', 'taylor', 'moore', 'jackson', 'martin'
            }
        }
        
        # Regex patterns for different PII types
        self.patterns = {
            PIIType.EMAIL: [
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.95),
            ],
            PIIType.PHONE: [
                (r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', 0.9),
                (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 0.85),
                (r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}', 0.9),
            ],
            PIIType.SSN: [
                (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', 0.8),
                (r'\b\d{9}\b', 0.6),  # Lower confidence for 9 digits alone
            ],
            PIIType.CREDIT_CARD: [
                # Visa
                (r'\b4[0-9]{12}(?:[0-9]{3})?\b', 0.9),
                # MasterCard
                (r'\b5[1-5][0-9]{14}\b', 0.9),
                # American Express
                (r'\b3[47][0-9]{13}\b', 0.9),
                # Discover
                (r'\b6(?:011|5[0-9]{2})[0-9]{12}\b', 0.9),
                # Generic 13-19 digits
                (r'\b(?:\d{4}[-.\s]?){3}\d{1,4}\b', 0.7),
            ],
            PIIType.IP_ADDRESS: [
                (r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 0.9),
                (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', 0.9),  # IPv6
            ],
            PIIType.DATE_OF_BIRTH: [
                (r'\b(?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2}\b', 0.8),
                (r'\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01])\b', 0.8),
            ],
            PIIType.PASSPORT: [
                (r'\b[A-Z]{1,2}[0-9]{6,9}\b', 0.7),  # US passport format
            ],
            PIIType.DRIVER_LICENSE: [
                (r'\b[A-Z]{1,2}[0-9]{6,8}\b', 0.6),  # Generic format
            ],
            PIIType.BANK_ACCOUNT: [
                (r'\b[0-9]{8,17}\b', 0.5),  # Generic bank account
            ],
            PIIType.URL: [
                (r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?', 0.95),
            ],
        }
    
    def detect_pii(self, text: str, context: str = "") -> List[PIIMatch]:
        """Detect PII in text"""
        matches = []
        
        # Pattern-based detection
        for pii_type in PIIType:
            if pii_type.value not in self.enabled_types:
                continue
                
            if pii_type in self.patterns:
                type_matches = self._detect_pattern_pii(text, pii_type, context)
                matches.extend(type_matches)
        
        # Name detection (more complex)
        if PIIType.NAME.value in self.enabled_types:
            name_matches = self._detect_names(text, context)
            matches.extend(name_matches)
        
        # Address detection
        if PIIType.ADDRESS.value in self.enabled_types:
            address_matches = self._detect_addresses(text, context)
            matches.extend(address_matches)
        
        # Remove overlapping matches (keep highest confidence)
        matches = self._remove_overlaps(matches)
        
        # Filter by minimum confidence
        matches = [m for m in matches if m.confidence >= self.min_confidence]
        
        return matches
    
    def _detect_pattern_pii(self, text: str, pii_type: PIIType, context: str) -> List[PIIMatch]:
        """Detect PII using regex patterns"""
        matches = []
        patterns = self.patterns.get(pii_type, [])
        
        for pattern, base_confidence in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Additional validation for some types
                confidence = base_confidence
                matched_text = match.group()
                
                if pii_type == PIIType.CREDIT_CARD:
                    confidence = self._validate_credit_card(matched_text, base_confidence)
                elif pii_type == PIIType.SSN:
                    confidence = self._validate_ssn(matched_text, base_confidence)
                elif pii_type == PIIType.IP_ADDRESS:
                    confidence = self._validate_ip_address(matched_text, base_confidence)
                
                if confidence >= self.min_confidence:
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        text=matched_text,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=context
                    ))
        
        return matches
    
    def _detect_names(self, text: str, context: str) -> List[PIIMatch]:
        """Detect person names"""
        matches = []
        
        # Look for capitalized words that might be names
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        
        for match in re.finditer(name_pattern, text):
            matched_text = match.group()
            words = matched_text.split()
            
            # Skip single words unless they're common names
            if len(words) == 1:
                if words[0].lower() not in self.common_names['first_names']:
                    continue
                confidence = 0.6
            else:
                # Multiple words - check if they look like names
                confidence = self._calculate_name_confidence(words)
            
            if confidence >= self.min_confidence:
                matches.append(PIIMatch(
                    pii_type=PIIType.NAME,
                    text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    context=context
                ))
        
        return matches
    
    def _calculate_name_confidence(self, words: List[str]) -> float:
        """Calculate confidence that words represent a person's name"""
        if not words:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check against common names
        for word in words:
            word_lower = word.lower()
            if word_lower in self.common_names['first_names']:
                confidence += 0.2
            elif word_lower in self.common_names['last_names']:
                confidence += 0.15
        
        # Penalize very long sequences
        if len(words) > 4:
            confidence -= 0.2
        
        # Boost for typical name patterns
        if len(words) == 2:  # First Last
            confidence += 0.1
        elif len(words) == 3:  # First Middle Last
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _detect_addresses(self, text: str, context: str) -> List[PIIMatch]:
        """Detect postal addresses"""
        matches = []
        
        # Simple address patterns
        address_patterns = [
            (r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court)\b', 0.8),
            (r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Rd|Road)\s*,?\s*[A-Z][a-z]+\s*,?\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', 0.9),
        ]
        
        for pattern, confidence in address_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(PIIMatch(
                    pii_type=PIIType.ADDRESS,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    context=context
                ))
        
        return matches
    
    def _validate_credit_card(self, text: str, base_confidence: float) -> float:
        """Validate credit card number using Luhn algorithm"""
        # Remove non-digits
        digits = re.sub(r'\D', '', text)
        
        if len(digits) < 13 or len(digits) > 19:
            return 0.0
        
        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0
        
        if luhn_check(digits):
            return base_confidence
        else:
            return max(0.3, base_confidence - 0.4)  # Lower confidence for invalid
    
    def _validate_ssn(self, text: str, base_confidence: float) -> float:
        """Validate SSN format"""
        digits = re.sub(r'\D', '', text)
        
        if len(digits) != 9:
            return 0.0
        
        # Check for invalid SSN patterns
        if digits == '000000000' or digits[:3] == '000' or digits[3:5] == '00' or digits[5:] == '0000':
            return 0.2
        
        return base_confidence
    
    def _validate_ip_address(self, text: str, base_confidence: float) -> float:
        """Validate IP address"""
        try:
            import ipaddress
            ipaddress.ip_address(text)
            return base_confidence
        except ValueError:
            return 0.0
    
    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence"""
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda m: m.start)
        
        result = []
        for match in matches:
            # Check if this match overlaps with any in result
            overlaps = False
            for i, existing in enumerate(result):
                if (match.start < existing.end and match.end > existing.start):
                    # Overlap detected
                    if match.confidence > existing.confidence:
                        # Replace existing with higher confidence match
                        result[i] = match
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(match)
        
        return result
    
    def redact_pii(self, text: str, matches: Optional[List[PIIMatch]] = None) -> Tuple[str, List[PIIMatch]]:
        """Redact PII from text"""
        if matches is None:
            matches = self.detect_pii(text)
        
        if not matches:
            return text, matches
        
        # Sort matches by start position (reverse order for replacement)
        matches.sort(key=lambda m: m.start, reverse=True)
        
        redacted_text = text
        for match in matches:
            replacement = self._create_replacement(match)
            redacted_text = redacted_text[:match.start] + replacement + redacted_text[match.end:]
        
        return redacted_text, matches
    
    def _create_replacement(self, match: PIIMatch) -> str:
        """Create replacement text for PII"""
        if self.preserve_format:
            # Preserve format for some types
            if match.pii_type == PIIType.EMAIL:
                parts = match.text.split('@')
                if len(parts) == 2:
                    return f"{'*' * len(parts[0])}@{parts[1]}"
            elif match.pii_type == PIIType.PHONE:
                # Preserve phone format
                digits_only = re.sub(r'\D', '', match.text)
                format_chars = re.sub(r'\d', '', match.text)
                replacement = self.redaction_char * len(digits_only)
                # Reconstruct with original formatting
                result = ""
                digit_idx = 0
                for char in match.text:
                    if char.isdigit():
                        result += self.redaction_char
                        digit_idx += 1
                    else:
                        result += char
                return result
            elif match.pii_type == PIIType.CREDIT_CARD:
                # Show last 4 digits
                digits = re.sub(r'\D', '', match.text)
                if len(digits) >= 4:
                    masked = self.redaction_char * (len(digits) - 4) + digits[-4:]
                    # Preserve original formatting
                    result = ""
                    digit_idx = 0
                    for char in match.text:
                        if char.isdigit():
                            result += masked[digit_idx] if digit_idx < len(masked) else self.redaction_char
                            digit_idx += 1
                        else:
                            result += char
                    return result
        
        # Default: replace with asterisks
        return self.redaction_char * len(match.text)
    
    def analyze_pii_risk(self, text: str) -> Dict:
        """Analyze PII risk in text"""
        matches = self.detect_pii(text)
        
        risk_scores = {
            PIIType.SSN: 10,
            PIIType.CREDIT_CARD: 9,
            PIIType.PASSPORT: 8,
            PIIType.DRIVER_LICENSE: 7,
            PIIType.BANK_ACCOUNT: 8,
            PIIType.EMAIL: 5,
            PIIType.PHONE: 6,
            PIIType.ADDRESS: 7,
            PIIType.DATE_OF_BIRTH: 6,
            PIIType.NAME: 4,
            PIIType.IP_ADDRESS: 3,
            PIIType.URL: 2,
        }
        
        total_risk = 0
        type_counts = {}
        
        for match in matches:
            risk = risk_scores.get(match.pii_type, 1) * match.confidence
            total_risk += risk
            
            if match.pii_type not in type_counts:
                type_counts[match.pii_type] = 0
            type_counts[match.pii_type] += 1
        
        # Normalize risk score
        max_possible_risk = len(matches) * 10 if matches else 1
        normalized_risk = min(1.0, total_risk / max_possible_risk)
        
        return {
            "total_matches": len(matches),
            "risk_score": normalized_risk,
            "risk_level": self._get_risk_level(normalized_risk),
            "pii_types_found": [t.value for t in type_counts.keys()],
            "type_counts": {t.value: count for t, count in type_counts.items()},
            "matches": [match.to_dict() for match in matches]
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level based on score"""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def get_statistics(self) -> Dict:
        """Get PII detector statistics"""
        return {
            "enabled_types": list(self.enabled_types),
            "total_patterns": sum(len(patterns) for patterns in self.patterns.values()),
            "min_confidence": self.min_confidence,
            "preserve_format": self.preserve_format,
            "common_names_loaded": len(self.common_names['first_names']) + len(self.common_names['last_names'])
        }


# Global PII detector instance
pii_detector: Optional[PIIDetector] = None


def get_pii_detector(config: Optional[Dict] = None) -> PIIDetector:
    """Get global PII detector instance"""
    global pii_detector
    if not pii_detector:
        pii_detector = PIIDetector(config)
    return pii_detector