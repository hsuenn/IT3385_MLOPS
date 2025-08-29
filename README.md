# IT3385 Machine Learning Operations Assignment

Models and their respective team members
| Name | Admin number | Model |
| ---- | ------------ | ----- |
| Choo Tze Hsuen | 220926F | [Wheat Type (Classification)](/src/streamlit/pages/1_Wheat.py) |
| Ian Chia Bing Jun | 230746D | [Melbourne Residential Prices (Regression)](/src/streamlit/pages/2_Melbourne.py) |
| Muhammad Aniq Sufi Bin Ismail | 2232237W | [Used Car Prices (Regression)](/src/streamlit/pages/3_Used_Car_Prices.py) |


# Startup

The project is packaged and managed by [peotry](https://python-poetry.org/).
This project offers multiple ways to get started.
| Method |   |
| ------ | - |
| Build from source | [Source setup](#source-setup) |
| Docker | [Docker setup](#docker-setup) |

## Source Setup
> [!NOTE]
> A terminal/shell session with Python binary is needed.
> This can be in the form of `py`, `python` or `python3`, depending on your operating system and Python installation method.

The project is packaged and managed by [poetry](https://python-poetry.org/).

To install poetry (on Windows), as per [official doc](https://python-poetry.org/docs/#installing-with-the-official-installer):
> [!TIP]
> The following instructions reference the Python binary as `py`, switch out for `python` if your binary is named `python` instead (i.e. Python installed via Microsoft Store has a separate executable name than the one installed with the official installer).
```
# recommend via pipx, install via pip if not yet installed
py -m pip install --user pipx

# install poetry with pipx (recommended by poetry)
pipx install poetry
```

> [!TIP]
> If `pipx` is still not recognised, please see [Troubleshooting Common Problems](#troubleshooting-common-problems). Otherwise, you may refer to their [official installation guide](https://pipx.pypa.io/stable/installation/).


To install poetry (for Linux/MacOS), as per [official doc](https://python-poetry.org/docs/#installing-with-the-official-installer):
```
curl -sSL https://install.python-poetry.org | python -
```
<hr>

Install the project either with `git` or download the zip file archive from this site directly.
```
git clone https://github.com/hsuenn/IT3385_MLOPS
```

Alternatively, you may download the latest zipped relesaes at [/Release]()
<hr>

After getting poetry installed, activate the environment (on Windows):
```
# navigate into the directory
cd IT3385_MLOPS
poetry install
$(poetry env activate)
```

After getting poetry installed, activate the environment (on Linux/MacOS):
```
# navigate into the directory
cd IT3385_MLOPS
poetry install
$(poetry env activate)
```
<hr>

To verify the environment have been activated successfully (on Windows):
```
poetry env info
which python
```

To verify the environment have been activated successfully (on Linux/MacOs):
```
poetry env info
where python
```
The `Executable` path shown by `poetry env info` should be exactly the same as path returned by `where python`, which shows where the current Python binary is stored at.
<hr>

After activating the environment, to start the streamlit server for UI inference.
```
python -m streamlit run src/streamlit/Home.py
```

The frotend server address is outputted to terminal (e.g. `http://localhost:8501`, your browser should automatically load up the page, otherwise manually keying in the address works too).

For common troubleshooting, please see [Troubleshooting Common Problems](#troubleshooting-common-problems).

## Docker Setup
Pull image from official docker repository.
```
# pull the image
docker pull tzehsuen/IT3385_MLOPS

# spin up a container
docker run tzehsuen/IT3385_MLOPS -p 6333:6333
```

Alternatively, you may build the docker image yourself.<br>
To do so, you will need clone the repository (in order to get the `Dockerfile`).
```
docker build . --tags=tzehsuen/IT3385_MLOPS
```


# Project Structure
The project follows the below structure.
```
IT3385_MLOPS
|-- pyproject.toml
|-- configs/
|   |-- training.yaml
|   |-- model.yaml
```

Config file stores the default values for most interface

Model training code is found in `src/models/*/train.py`


# Troubleshooting Common Problems

> `pipx` is not recognized as an internal or external command, operable program, or batch file

Here are some potential resolution for installing `pipx` on Windows
- Ensure it's install globally with the `--user` (or `-U`) flag (e.g. `python -m pip install --user pipx`)
- Ensure `pip` version is 19.0 or later (as specified in their [official installation guide](https://pipx.pypa.io/stable/installation/))
- Restart your terminal session to reload your shell profile so that the `pipx` executable path is included in your search path (e.g. close and open a new terminal window)

<hr>

> OSError: dlopen(/pypoetry/virtualenvs/it3385-mlops-arMP4vTn-py3.11/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib

On MacOS, you will need to install `libomp` as solved by a user here on [stackoverflow](https://stackoverflow.com/a/55958281)

```
brew install libomp
```

