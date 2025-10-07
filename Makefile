install:
	pip install -r requirements.txt

lint:
	pre-commit run --all-files

api-deploy:
	uvicorn scripts.main:app --host 0.0.0.0 --port 8080 --reload

build-rag:
	docker compose up --build

test:
	python ./scripts/test.py

run-prefect:
	prefect server start

run-streamlit:
	streamlit run ./streamlit_app/app.py
