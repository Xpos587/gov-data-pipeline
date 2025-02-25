project_dir := "."
venv_prefix := "micromamba run -p ./.micromamba"

lint:
	{{venv_prefix}} black --check --diff {{project_dir}}
	{{venv_prefix}} ruff check {{project_dir}}
	{{venv_prefix}} mypy {{project_dir}} --strict

reformat:
	{{venv_prefix}} black {{project_dir}}
	{{venv_prefix}} ruff format {{project_dir}}

run:
	{{venv_prefix}} python main.py || true

build-container:
  podman build -f Containerfile -t gov_data_pipeline .
