import argilla as rg
from dotenv import load_dotenv
import os
from datasets import load_dataset
from argilla.markdown import audio_to_html
import soundfile as sf
from tqdm import tqdm


def prepare_audio_for_argilla(example):
    """Prepares audio data for logging to Argilla using audio_to_html."""
    name=example["audio"]["path"]
    audio_array = example["audio"]["array"]
    sample_rate = example["audio"]["sampling_rate"]

    sf.write(name, audio_array, sample_rate)


    # Generate HTML using audio_to_html
    audio_html = audio_to_html(name,width="300px",
    height="300px",
    autoplay=True,
    loop=True) 
    #os.remove("output_audio.mp3")

    return {"audio": audio_html, "transcription": example["transcription"]} 

if __name__ == "__main__":
    # Load environment variables (make sure your .env file contains ARGILLA_API_KEY)
    load_dotenv()
    API_KEY = os.environ["ARGILLA_KEY"]
    HF_API_KEY = os.environ["HF_API_KEY"]

    # Initialize Argilla with your API key
    rg.Argilla.deploy_on_spaces(api_key=API_KEY)
    dataset=load_dataset("atlasia/Youtube-Commons-Morocco-Darija-35h",split="train",token=HF_API_KEY).select_columns(["audio","transcription"])
    #dataset=dataset.map(prepare_audio_for_argilla)
    processed_records = []
    for i, example in tqdm(enumerate(dataset),total=len(dataset)):
        example['id'] = i  # Add an ID to make unique filenames
        processed_record = prepare_audio_for_argilla(example)
        processed_records.append(processed_record)

    setting=rg.Settings(
        fields=[
            rg.TextField(
                        name="audio",
                        use_markdown=True,
                        required=True
                    )
        ],
        questions=[
            rg.TextQuestion(name="Transcription",
                            title="Correct The Transcription:",
                            use_markdown=True, required=True)],
    )
    dataset_arg=rg.Dataset(
        "Transcription Alignment",
        settings=setting)
    dataset_arg.create()
    dataset_arg.records.log(processed_records, mapping={"audio": "audio", "transcription":"Transcription"})

    