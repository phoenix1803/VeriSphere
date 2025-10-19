# VeriSphere - Multi-Agent Misinformation Detection System

VeriSphere is a sophisticated, production-ready misinformation detection system that employs multiple AI agents to verify claims through a comprehensive six-stage pipeline. The system combines machine learning, knowledge verification, logical analysis, and real-time web scraping to provide accurate, explainable fact-checking results.

## Features

### Core Capabilities
- **Multi-Agent Architecture**: Specialized agents for different verification approaches
- **Six-Stage DEFAME Pipeline**: Detection, Evidence, Fact-checking, Analysis, Mitigation, Evaluation
- **Real-Time Processing**: Fast claim verification with configurable priority levels
- **Multimodal Support**: Text, image, and combined content verification
- **Explainable Results**: Detailed reasoning and evidence for all verdicts

### Advanced Features
- **MCP Orchestration**: Intelligent agent coordination and consensus building
- **Blockchain Audit Trail**: Immutable verification records
- **Crisis Communication**: Automated alerts for high-impact misinformation
- **Batch Processing**: Handle multiple claims efficiently
- **Web Dashboard**: User-friendly interface for claim submission and monitoring
- **Authentication & Authorization**: JWT-based security with role-based access
- **Rate Limiting**: Advanced rate limiting with multiple strategies
- **PII Detection**: Automatic detection and redaction of sensitive information

### Security & Production Features
- **Advanced Rate Limiting**: Per-IP, per-user, and global rate limits
- **PII Detection & Redaction**: Automatic handling of sensitive data
- **Comprehensive Authentication**: JWT tokens, API keys, role-based permissions
- **Audit Logging**: Complete audit trail for compliance
- **Monitoring & Metrics**: Prometheus-compatible metrics and health checks
- **Kubernetes Deployment**: Production-ready container orchestration
- **CI/CD Pipeline**: Automated testing, building, and deployment

## Architecture

### Agents
1. **ML Agent**: Deep learning-based verification using transformer models
2. **Wikipedia Agent**: Knowledge base verification and cross-referencing
3. **Coherence Agent**: Logical consistency and contradiction detection
4. **WebScrape Agent**: Real-time web information gathering and analysis

### Evidence Tools
- **Search Tool**: Web search using Serper API
- **Image Tool**: Google Vision API integration for image analysis
- **Firecrawl Scraper**: Advanced web content extraction
- **Geolocation Tool**: Location-based claim verification

### Security Components
- **Rate Limiter**: Redis-backed advanced rate limiting
- **PII Detector**: Pattern-based PII detection and redaction
- **Auth System**: JWT authentication with role-based permissions
- **Audit Logger**: Comprehensive audit trail system

## Requirements

- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Docker (optional)
- Kubernetes (for production)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/verisphere.git
cd verisphere

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Required API keys:
- `HUGGINGFACE_API_KEY`: For ML models
- `SERPER_API_KEY`: For web search
- `GOOGLE_VISION_API_KEY`: For image analysis
- `FIRECRAWL_API_KEY`: For web scraping

### 3. Database Setup

```bash
# Start PostgreSQL and Redis (using Docker)
docker-compose up -d postgres redis

# Initialize database
python -c "from defame.core.database import init_database; init_database()"
```

### 4. Run the System

```bash
# Start API server
python scripts/run_api.py

# Or run with specific configuration
python scripts/run_config.py config/sample_config.yaml

# Test the system
python test_system.py
```

### 5. Access the Dashboard

Open your browser and navigate to `http://localhost:8000` to access the web dashboard.

Default admin credentials:
- Username: `admin`
- Password: `admin123`

## Usage

### Web Dashboard

1. **Login**: Navigate to `http://localhost:8000/login`
2. **Submit Claims**: Use the "Submit Claim" tab to verify content
3. **Monitor Progress**: Track claim processing in real-time
4. **View Results**: Get detailed verification results with evidence

### API Examples

#### Authentication
```bash
# Login to get JWT token
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  "http://localhost:8000/api/v1/status"
```

#### Submit a Claim
```bash
curl -X POST "http://localhost:8000/api/v1/claims" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "content": "The Earth is flat",
    "claim_type": "text",
    "priority": "normal",
    "source": "social_media"
  }'
```

#### Batch Processing
```bash
curl -X POST "http://localhost:8000/api/v1/batch/jobs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "name": "Daily fact-check batch",
    "description": "Process daily claims",
    "claims": [
      {"content": "Claim 1", "claim_type": "text"},
      {"content": "Claim 2", "claim_type": "text"}
    ]
  }'
```

#### File Upload
```bash
curl -X POST "http://localhost:8000/api/v1/claims/upload" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@image.jpg" \
  -F "claim_type=image" \
  -F "priority=high"
```

### Python SDK

```python
from defame.core.models import Claim
from defame.core.pipeline import get_pipeline_controller
from config.globals import ClaimType, Priority

# Create a claim
claim = Claim(
    content="Vaccines cause autism",
    claim_type=ClaimType.TEXT,
    priority=Priority.HIGH
)

# Process the claim
pipeline = get_pipeline_controller()
result = await pipeline.process_claim(claim)

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.overall_confidence}")
```

## Docker Deployment

### Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# Scale workers
docker-compose up --scale worker=3
```

### Production
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Or use Kubernetes
kubectl apply -f k8s/
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Ingress controller (NGINX recommended)
- cert-manager (for TLS certificates)

### Deployment Steps

```bash
# Create namespace and apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/monitoring.yaml

# Check deployment status
kubectl get pods -n verisphere
kubectl get services -n verisphere
kubectl get ingress -n verisphere
```

### Monitoring
```bash
# View logs
kubectl logs -f deployment/verisphere-api -n verisphere

# Check metrics
kubectl port-forward service/verisphere-api-internal 8000:8000 -n verisphere
curl http://localhost:8000/metrics
```

## Monitoring & Observability

### Health Checks
- API: `GET /health`
- System Status: `GET /api/v1/status`
- Metrics: `GET /metrics` (Prometheus format)

### Prometheus Metrics
- `verisphere_claims_processed_total`: Total claims processed
- `verisphere_average_processing_time_seconds`: Average processing time
- `verisphere_accuracy_score`: System accuracy score
- `verisphere_healthy_agents`: Number of healthy agents
- `verisphere_batch_jobs_total`: Total batch jobs

### Grafana Dashboard
Import the dashboard configuration from `k8s/monitoring.yaml` for comprehensive system monitoring.

## Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=defame --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Performance Testing
```bash
# Load testing with Locust
pip install locust
locust -f tests/performance/load_test.py --host=http://localhost:8000
```

### Security Testing
```bash
# Security scan
bandit -r defame/

# Dependency check
safety check
```

## Performance

### Benchmarks
- **Text Claims**: ~2-5 seconds average processing time
- **Image Claims**: ~10-15 seconds average processing time
- **Throughput**: 100+ claims per minute (with proper scaling)
- **Accuracy**: 85-92% on standard fact-checking datasets

### Scaling Guidelines
- **API Servers**: 3-20 replicas based on load
- **Workers**: 2-50 replicas based on queue depth
- **Database**: Use read replicas for high read loads
- **Redis**: Use Redis Cluster for high availability

## Security

### Authentication & Authorization
- JWT-based authentication with configurable expiration
- Role-based access control (admin, user, api_user, readonly)
- API key authentication for service-to-service communication
- Secure password hashing with bcrypt

### Rate Limiting
- Per-IP, per-user, and per-API-key rate limits
- Configurable limits for different endpoints
- Redis-backed sliding window algorithm
- Automatic cleanup of expired entries

### PII Protection
- Automatic detection of emails, phone numbers, SSNs, credit cards
- Configurable redaction with format preservation
- Risk assessment and compliance reporting
- Support for custom PII patterns

### Security Headers
- CORS configuration
- Security headers (CSP, X-Frame-Options, etc.)
- Request size limits
- Input validation and sanitization


### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
flake8 .
mypy defame/

# Run tests
pytest tests/
```

### Code Quality Standards
- 90%+ test coverage required
- All code must pass linting (black, flake8, mypy)
- Security scan must pass (bandit)
- Documentation required for new features

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions pipeline:

- **Testing**: Unit, integration, and security tests
- **Quality**: Code formatting, linting, and type checking
- **Security**: Vulnerability scanning and dependency checks
- **Building**: Multi-platform Docker images
- **Deployment**: Automated staging and production deployments
- **Monitoring**: Performance testing and health checks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
