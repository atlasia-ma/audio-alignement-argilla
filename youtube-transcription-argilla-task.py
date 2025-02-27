import os
import argilla as rg
from datasets import load_dataset
import soundfile as sf
from argilla.markdown import audio_to_html
from tqdm import tqdm
from dotenv import load_dotenv
import yaml

def prepare_audio_for_argilla(example):
    """
    Prepares audio data for logging to Argilla by writing the audio to a file,
    converting it to an HTML snippet using `audio_to_html`, and returning the processed record.
    """
    record_id = example["id"] # created in the map function before calling this function
    audio_info = example["audio"]
    
    audio_array = audio_info["array"]
    sample_rate = audio_info["sampling_rate"]
    
    # create directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)
    
    # Use a unique filename for each audio record
    file_name = f"./tmp/audio_{record_id}.mp3"
    
    # Write the audio data to a file if it doesn't already exist
    if not os.path.exists(file_name):
        sf.write(file_name, audio_array, sample_rate)
    
    # Generate an HTML snippet to embed the audio player
    audio_html = audio_to_html(
        file_name,
        width="300px",
        height="300px",
        autoplay=True,
        loop=False,
    )
  
    return {
        "id": record_id,
        "audio": audio_html,
        "original_transcription": example["transcription"],
        "corrected_transcription": example["transcription"]  # default value prefilled
    }

if __name__ == "__main__":
    
    # Load env variables
    load_dotenv()
    ARGILLA_KEY = os.environ["ARGILLA_KEY"]
    ARGILLA_API_URL = os.environ["ARGILLA_API_URL"]
    HF_TOKEN = os.environ["HF_TOKEN"]

    # load config file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the Argilla client
    client = rg.Argilla(
        api_url=ARGILLA_API_URL,
        api_key=ARGILLA_KEY,
        headers={"Authorization": f"Bearer {HF_TOKEN}"} # required if space is private
    )

    settings = rg.Settings(
        guidelines=config['ANNOTATION_GUIDELINES'],
        fields=[
            rg.TextField(name="audio", use_markdown=True, required=True),
            rg.TextField(name="original_transcription", required=True),
        ],
        questions=[
            rg.RatingQuestion(
                name="transcription_quality",
                title="How good is the transcription? \n(0: Incorrect, 1: Partially Correct, 2: Totally Correct)",
                values=[0, 1, 2],
                required=False
            ),
            rg.TextQuestion(
                name="corrected_transcription",
                title="Provide the correct transcription if needed (only required if rating is 0 or 1)",
                required=True
            ),
            rg.LabelQuestion(
                name="with_code_switching",
                title="Code Switching (MSA not counted)",
                labels=["True", "False"],
                required=True
            ),
            rg.LabelQuestion(
                name="with_msa",
                title="Includes MSA",
                labels=["True", "False"],
                required=True
            ),
            rg.LabelQuestion(
                name="with_pauses_hesitations",
                title="Pauses/hesitations presence (um, uh, etc.)",
                labels=["True", "False"],
                required=True
            ),
            rg.LabelQuestion(
                name="speaker_gender",
                title="Speaker Gender",
                labels=["Male", "Female"],
                required=True
            ),
            rg.LabelQuestion(
                name="emotion_sentiment",
                title="Emotion / Sentiment",
                labels=["Very Positive", "Positive", "Neutral", "Negative", "Very Negative", "Other"],
                required=False
            ),
            rg.LabelQuestion(
                name="background_noise",
                title="Background Noise Level",
                labels=["None", "Low", "Moderate", "High"],
                required=False
            ),
            rg.LabelQuestion(
                name="intent",
                title="Utterance Intent",
                labels=["Question", "Informative/Statement", "Command", "Other"],
                required=False
            ),
            rg.LabelQuestion(
                name="speaker_clarity",
                title="Speaker Clarity",
                labels=["Clear", "Somewhat Clear", "Muffled/Unclear"],
                required=False
            ),
            rg.LabelQuestion(
                name="speaker_count",
                title="Speaker Count",
                labels=["Single Speaker", "Multiple Speakers"],
                required=False
            ),
        ],
    )
    
    # loop through all datasets to use
    for hf_dataset_name in config['dataset_list']:
        # get dataset name
        argilla_dataset_name = hf_dataset_name.split("/")[-1]
        # load dataset split
        hf_dataset = load_dataset(
            hf_dataset_name,
            split=config['UPLOAD_SPLIT'],
            token=HF_TOKEN
        )
        # prepare dataset
        hf_dataset = hf_dataset.map(lambda _, idx: {"id": idx}, with_indices=True, desc="Adding record IDs")
        # create Argilla dataset       
        dataset = rg.Dataset(
            name=f"{argilla_dataset_name}",
            settings=settings,
            client=client,
        )
        dataset.create()
        
        # prepare data for Argilla
        processed_records = []
        for example in tqdm(hf_dataset, total=len(hf_dataset), desc=f"Preparing data for Argilla: {argilla_dataset_name}"):
            processed_record = prepare_audio_for_argilla(example)
            processed_records.append(processed_record)
            
        # put data in records
        dataset.records.log(
            records=processed_records,
            mapping={
                "audio": "audio",
                "original_transcription": "original_transcription",
                "corrected_transcription": "corrected_transcription"
            }
        )
        
        print(f"[INFO] Successfully pushed {len(processed_records)} records to Argilla for dataset: {argilla_dataset_name}")
    
    print("[INFO] All specified datasets have been pushed to Argilla with the defined settings.")


