# Aeneas Text-to-Speech Alignment

This project provides a Modal service that uses the Aeneas library to align text with audio files. The service accepts a text input and audio bytes, and returns the alignment information in JSON format.

## Installation

To install the dependencies using uv:

```bash
uv sync
```

## Prerequisites

- Python >= 3.10
- A [modal.com](https://www.modal.com/) account
- Modal Python SDK >= 1.2

## Deployment

Setup the Modal CLI:

```bash
modal setup
```

Deploy the service:

```bash
modal deploy app.py
```

### Development & Local Testing

To test locally:

```bash
modal run app.py
```

## Calling via Modal SDK

### Python SDK

```python
import modal

Model = modal.Cls.from_name("aeneas", "Model")
aligned_text = Model().inference.remote(text="Your text here", audio_file=audio_bytes)
```

### JavaScript / TypeScript SDK

```javascript
import { ModalClient } from "@modal-labs/client";

const client = new ModalClient();
const fn = await client.functions.get("aeneas", "Model.inference");
const result = await fn.call({
  text: "Your text here",
  audio_file: audioBuffer
});
```

### Testing Deployed Service

Run the test client with the sample audio and text from `api.rest`:

```bash
python client.py
```
