.PHONY: install lint format test run up down logs pull-models loadtest

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

# Run the gateway on the host (needs Redis, or REDIS_URL=memory://).
run:
	uvicorn llm_gateway.app:app_from_env --factory --reload --port 8080

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f gateway

pull-models:
	docker compose run --rm ollama-pull

loadtest:
	docker run --rm -i --network host -e BASE_URL=http://localhost:8080 grafana/k6 run - < loadtest/k6-chat.js
