# Aeneas Text-to-Speech Alignment

This project provides a FastAPI-based web service that uses the Aeneas library to align text with audio files. The service accepts a text input and an audio file, and returns the alignment information in JSON format. The service is built on the [Modal.com](https://www.modal.com/) platform.

## Installation

To install the project, clone the repository and install the dependencies using pip:

```bash
git clone git@github.com:FelippeChemello/modal_aeneas.git
cd modal_aeneas
pip install -r requirements.txt
```

## Prerequisites

- Python 3.10
- A [modal.com](https://www.modal.com/) account
- An `API_KEY` for the service, stored as a secret in Modal.com under the name `aeneas-secret`


## Deployment

to run the service you need to setup modal.com CLI

```bash
modal setup
```

and then deploy the service

```bash
modal deploy app.py --name aeneas
```

### Development

To run the service locally, use the following command:

```bash
modal serve app.py 
```

## Usage

To use the service, send a POST request to the root URL, provided by modal, with the following parameters as form data:

- `text`: The text to align with the audio file
- `audio_file`: The audio file to align with the text

along with the `API_KEY` in the headers as `x-api-key`

The service will return a JSON response with the alignment information.



