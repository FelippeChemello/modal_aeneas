import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "espeak",
        "espeak-data",
        "libespeak1",
        "libespeak-dev",
        "festival*",
        "build-essential",
        "flac",
        "libasound2-dev",
        "libsndfile1-dev",
        "vorbis-tools",
        "libxml2-dev",
        "libxslt-dev",
        "zlib1g-dev",
    )
    .uv_pip_install(
        "numpy==1.24.2",
        "packaging==23.1",
        "pandas==1.5.3",
        "psycopg2-binary==2.9.6",
        "platformdirs==2.6.0",
        "plac==1.3.5",
        "pipenv==2022.11.30",
        "parsel==1.8.1",
        "Protego==0.2.1",
        "prompt-toolkit==3.0.38",
        "nvidia-cublas-cu11==11.10.3.66",
        "nvidia-cuda-nvrtc-cu11==11.7.99",
        "nvidia-cuda-runtime-cu11==11.7.99",
        "nvidia-cudnn-cu11==8.5.0.96",
        "curl_cffi",
        "cbor2",
    )
    .pip_install("aeneas", "cbor2")
)

app = modal.App("aeneas")

with image.imports():
    import json
    import tempfile
    from pathlib import Path
    from aeneas.executetask import ExecuteTask
    from aeneas.runtimeconfiguration import RuntimeConfiguration
    from aeneas.task import Task


@app.cls(gpu=None, image=image, timeout=150)
class Model:
    def _run_alignment(self, text: str, audio_file: bytes):
        print("Aligning text to speech...")
        audio_file_path = Path(tempfile.mktemp(suffix=".wav"))
        text_file_path = Path(tempfile.mktemp(suffix=".txt"))
        sync_file_path = Path(tempfile.mktemp(suffix=".json"))

        with open(text_file_path, "w") as f:
            f.write("\n".join(text.split()))

        with open(audio_file_path, "wb") as f:
            f.write(audio_file)

        config_string = "task_language=por|is_text_type=plain|os_task_file_level=3|os_task_file_format=json"
        rconf = RuntimeConfiguration()
        rconf[RuntimeConfiguration.MFCC_MASK_NONSPEECH] = "True"
        rconf[RuntimeConfiguration.MFCC_MASK_NONSPEECH_L3] = "True"

        task = Task(config_string=config_string)
        task.text_file_path_absolute = text_file_path
        task.audio_file_path_absolute = audio_file_path
        task.sync_map_file_path_absolute = sync_file_path

        ExecuteTask(task, rconf=rconf).execute()

        task.output_sync_map_file()

        with open(task.sync_map_file_path_absolute, "r") as f:
            sync_data = json.load(f)

        return sync_data

    @modal.method()
    def inference(self, text: str, audio_file: bytes):
        return self._run_alignment(text, audio_file)

    @modal.method()
    def _inference(self, text: str, audio_file: bytes):
        return self._run_alignment(text, audio_file)


@app.local_entrypoint()
def main():
    text = "Você sabia que testes de sangue revolucionários para diagnosticar Alzheimer podem estar disponíveis em breve? Isso poderá transformar a maneira como lidamos com uma das doenças mais desafiadoras da atualidade."
    audio_file = open("example.wav", "rb").read()
    aligned_text = Model().inference.remote(text, audio_file)

    with open("aligned_text.json", "w") as f:
        f.write(json.dumps(aligned_text, indent=4))
