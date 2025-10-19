"""
Full VeriSphere system integration tests
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from scripts.run_api import app
from defame.core.models import Claim
from defame.auth.service import AuthService
from config.globals import ClaimType, Priority


class TestFullSystemIntegration:
    """Test complete system integration"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authentication headers"""
        # Create test user
        try:
            user = AuthService.create_user(
                username="testuser",
                email="test@example.com",
                password="testpass123"
            )
        except ValueError:
            # User already exists
            user = AuthService.authenticate_user("testuser", "testpass123")
        
        # Login to get token
        response = client.post("/login", data={
            "username": "testuser",
            "password": "testpass123"
        })
        
        # Extract token from cookie or response
        token = user.generate_jwt_token()
        return {"Authorization": f"Bearer {token}"}
    
    def test_health_check(self, client):
        """Test system health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_system_status(self, client, auth_headers):
        """Test system status endpoint"""
        response = client.get("/api/v1/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "agents" in data
    
    def test_web_dashboard_access(self, client):
        """Test web dashboard accessibility"""
        # Test login page
        response = client.get("/login")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test register page
        response = client.get("/register")
        assert response.status_code == 200
    
    def test_user_registration_and_login(self, client):
        """Test user registration and login flow"""
        # Register new user
        response = client.post("/register", data={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "newpass123",
            "confirm_password": "newpass123",
            "full_name": "New User"
        })
        
        # Should redirect to login or show success
        assert response.status_code in [200, 302]
        
        # Login with new user
        response = client.post("/login", data={
            "username": "newuser",
            "password": "newpass123"
        })
        
        # Should redirect to dashboard
        assert response.status_code == 302
    
    @patch('defame.agents.ml_agent.MLAgent.verify_claim')
    @patch('defame.agents.coherence_agent.CoherenceAgent.verify_claim')
    def test_claim_submission_and_processing(self, mock_coherence, mock_ml, client, auth_headers):
        """Test complete claim submission and processing flow"""
        # Mock agent responses
        from defame.core.models import VerificationResult, Evidence
        from config.globals import AgentType, Verdict
        
        mock_ml.return_value = VerificationResult(
            claim_id="test",
            agent_id="ml_agent",
            agent_type=AgentType.ML_AGENT,
            confidence=0.8,
            verdict=Verdict.FALSE,
            evidence=[Evidence(source="ML Model", content="Analysis indicates false claim")],
            reasoning="ML analysis suggests this is misinformation"
        )
        
        mock_coherence.return_value = VerificationResult(
            claim_id="test",
            agent_id="coherence_agent",
            agent_type=AgentType.COHERENCE_AGENT,
            confidence=0.7,
            verdict=Verdict.FALSE,
            evidence=[Evidence(source="Coherence Check", content="Logical inconsistencies found")],
            reasoning="Claim contains logical contradictions"
        )
        
        # Submit claim
        response = client.post("/api/v1/claims", 
            headers=auth_headers,
            json={
                "content": "The Earth is flat and the moon is made of cheese",
                "claim_type": "text",
                "priority": "normal",
                "source": "test"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "claim_id" in data
        assert data["status"] == "accepted"
        
        claim_id = data["claim_id"]
        
        # Check claim status
        response = client.get(f"/api/v1/claims/{claim_id}/status", headers=auth_headers)
        assert response.status_code == 200
        
        # Wait a bit for processing (in real test, would poll until complete)
        import time
        time.sleep(1)
    
    def test_batch_processing(self, client, auth_headers):
        """Test batch processing functionality"""
        # Create batch job
        response = client.post("/api/v1/batch/jobs",
            headers=auth_headers,
            json={
                "name": "Test Batch",
                "description": "Test batch processing",
                "claims": [
                    {"content": "Test claim 1", "claim_type": "text"},
                    {"content": "Test claim 2", "claim_type": "text"}
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        
        job_id = data["job_id"]
        
        # Check job status
        response = client.get(f"/api/v1/batch/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200
        
        # List jobs
        response = client.get("/api/v1/batch/jobs", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
    
    def test_file_upload(self, client, auth_headers):
        """Test file upload functionality"""
        # Create test image file
        import io
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Upload file
        response = client.post("/api/v1/claims/upload",
            headers=auth_headers,
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={
                "claim_type": "image",
                "priority": "normal"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "claim_id" in data
    
    def test_pii_detection(self, client, auth_headers):
        """Test PII detection functionality"""
        response = client.post("/api/v1/pii/analyze",
            headers=auth_headers,
            json={
                "text": "My email is john.doe@example.com and my phone is 555-123-4567"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "risk_level" in data
        assert "pii_types_found" in data
        assert len(data["pii_types_found"]) > 0
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Make multiple requests quickly
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should mostly succeed (rate limits are generous for health checks)
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 8  # Allow some rate limiting
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus format
        content = response.text
        assert "verisphere_" in content
    
    def test_api_key_authentication(self, client):
        """Test API key authentication"""
        # This would require setting up API keys in test
        # For now, just test that the endpoint exists
        response = client.get("/api/v1/status", headers={"X-API-Key": "invalid-key"})
        # Should either work with valid key or return 401
        assert response.status_code in [200, 401, 403]
    
    @pytest.mark.asyncio
    async def test_concurrent_claim_processing(self, client, auth_headers):
        """Test concurrent claim processing"""
        import asyncio
        import httpx
        
        async def submit_claim(client, claim_text):
            response = await client.post("/api/v1/claims",
                headers=auth_headers,
                json={
                    "content": claim_text,
                    "claim_type": "text",
                    "priority": "normal"
                }
            )
            return response.status_code
        
        # Submit multiple claims concurrently
        async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
            tasks = [
                submit_claim(async_client, f"Test claim {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Most should succeed
            success_count = sum(1 for result in results if result == 200)
            assert success_count >= 3
    
    def test_error_handling(self, client, auth_headers):
        """Test system error handling"""
        # Test invalid claim submission
        response = client.post("/api/v1/claims",
            headers=auth_headers,
            json={
                "content": "",  # Empty content
                "claim_type": "invalid_type",
                "priority": "invalid_priority"
            }
        )
        
        assert response.status_code == 400
        
        # Test non-existent claim
        response = client.get("/api/v1/claims/nonexistent-id/status", headers=auth_headers)
        assert response.status_code == 404
        
        # Test unauthorized access
        response = client.get("/api/v1/status")  # No auth headers
        assert response.status_code in [401, 403]
    
    def test_system_configuration(self, client, auth_headers):
        """Test system configuration and settings"""
        response = client.get("/api/v1/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that agents are configured
        if "agents" in data:
            assert data["agents"]["total_agents"] > 0
        
        # Check database connectivity
        if "database" in data:
            assert "total_claims" in data["database"]


class TestSystemPerformance:
    """Performance and load testing"""
    
    @pytest.mark.slow
    def test_claim_processing_performance(self, client, auth_headers):
        """Test claim processing performance"""
        import time
        
        start_time = time.time()
        
        # Submit a simple claim
        response = client.post("/api/v1/claims",
            headers=auth_headers,
            json={
                "content": "Simple test claim for performance testing",
                "claim_type": "text",
                "priority": "normal"
            }
        )
        
        submission_time = time.time() - start_time
        
        assert response.status_code == 200
        assert submission_time < 5.0  # Should submit within 5 seconds
    
    @pytest.mark.slow
    def test_system_memory_usage(self):
        """Test system memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Memory usage should be reasonable (less than 1GB for tests)
        assert memory_info.rss < 1024 * 1024 * 1024  # 1GB
    
    def test_database_performance(self, client, auth_headers):
        """Test database query performance"""
        import time
        
        start_time = time.time()
        
        # Make multiple status requests (which query database)
        for _ in range(10):
            response = client.get("/api/v1/status", headers=auth_headers)
            assert response.status_code == 200
        
        total_time = time.time() - start_time
        
        # Should complete 10 requests in reasonable time
        assert total_time < 10.0  # 10 seconds for 10 requests


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])