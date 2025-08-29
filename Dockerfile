# builder stage
FROM python:3.11-slim

# set poetry behavior through
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# update essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# install pipx to install poetry
RUN python3 -m pip install --user pipx
ENV PATH="/root/.local/bin:$PATH"
RUN pipx ensurepath

# install poetry with pipx
RUN pipx install poetry

# copy over pyproject.toml and poetry.lock to run poetry install
WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# copy python binary and other binaries, alongside project files (./README.md is needed because of pyproject.toml reference, otherwise poetry install will fail)
COPY . .

# install packages, except for development packages
RUN poetry install --without dev


# run streamlit app on port 8501
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "./src/streamlit/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
