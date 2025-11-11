# Hugo Makefile
# Convenience commands for common operations

.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down

help:
	@echo "Hugo Development Commands"
	@echo "========================="
	@echo ""
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install development dependencies"
	@echo "  make test           Run test suite"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo "  make clean          Clean build artifacts"
	@echo "  make docker-build   Build Docker services"
	@echo "  make docker-up      Start Docker services"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make hugo-up        Start Hugo"
	@echo "  make hugo-down      Stop Hugo"
	@echo "  make hugo-shell     Enter Hugo shell"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

dev-install: install
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy isort

test:
	pytest tests/ -v --cov=core --cov=runtime --cov=data

lint:
	flake8 core/ runtime/ data/ skills/
	mypy core/ runtime/ data/

format:
	black core/ runtime/ data/ skills/ tests/
	isort core/ runtime/ data/ skills/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

docker-build:
	docker-compose -f configs/docker-compose.yaml build

docker-up:
	docker-compose -f configs/docker-compose.yaml up -d

docker-down:
	docker-compose -f configs/docker-compose.yaml down

docker-logs:
	docker-compose -f configs/docker-compose.yaml logs -f

hugo-up:
	python -m runtime.cli up

hugo-down:
	python -m runtime.cli down

hugo-shell:
	python -m runtime.cli shell

hugo-status:
	python -m runtime.cli status

# Database management
db-init:
	docker exec -i hugo-db psql -U hugo_user -d hugo < services/db/init.sql

db-backup:
	docker exec hugo-db pg_dump -U hugo_user hugo > data/backups/hugo_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@echo "Usage: make db-restore FILE=path/to/backup.sql"
	docker exec -i hugo-db psql -U hugo_user -d hugo < $(FILE)

# Development utilities
create-skill:
	@read -p "Skill name: " skill_name; \
	python -m runtime.cli skill --new $$skill_name

validate-skill:
	@read -p "Skill name: " skill_name; \
	python -m runtime.cli skill --validate $$skill_name

# Setup
setup: install
	@echo "Creating data directories..."
	mkdir -p data/memory data/reflections data/logs data/backups data/vault
	@echo "Copying environment template..."
	cp -n .env.example .env || true
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env with your API keys"
	@echo "  2. Run 'make docker-up' to start services"
	@echo "  3. Run 'make hugo-up' to start Hugo"
	@echo ""
