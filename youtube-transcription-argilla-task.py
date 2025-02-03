import os
import argilla as rg
from datasets import load_dataset
import soundfile as sf
from argilla.markdown import audio_to_html
from tqdm import tqdm

# Read the Hugging Face token
HF_TOKEN = "..."

# Define annotation guidelines with detailed descriptions for each field and question
ANNOTATION_GUIDELINES = """
Annotation Guidelines:

1. Transcription Quality:
   - Rate the quality of the transcription based on the following scale:
     • 0: Incorrect – The transcription does not accurately represent the spoken content.
     • 1: Partially Correct – The transcription is mostly correct but contains minor misspellings or errors.
     • 2: Totally Correct – The transcription is exactly as spoken.
   
2. Corrected Transcription:
   - If the transcription quality is rated as 0 or 1, please provide the corrected transcription.
   - This field is pre-filled with the original transcription. Only modify it if corrections are needed.
   
3. Speaker Gender:
   - Select the gender of the speaker as perceived from the audio.
   - Options: Male, Female.
   
4. Emotion / Sentiment:
   - Assess the emotional tone or sentiment expressed in the audio.
   - Options: Very Positive, Positive, Neutral, Negative, Very Negative, Other.
   
5. Utterance Intent:
   - Identify the intent behind the utterance.
   - Options: Question, Informative/Statement, Command, Other.
   
6. Background Noise Level:
   - Evaluate the level of background noise present in the audio.
   - Options: None, Low, Moderate, High.
   
7. Speaker Clarity:
   - Determine how clear the speaker’s voice is.
   - Options: Clear, Somewhat Clear, Muffled/Unclear.
   
8. Speaker Count:
   - Indicate whether the audio contains a single speaker or multiple speakers.
   - Options: Single Speaker, Multiple Speakers.

Please listen carefully to the audio and use the structured options to ensure consistent annotations.
"""

# Initialize the Argilla client
client = rg.Argilla(
    api_url="...",
    api_key="...",
    headers={"Authorization": f"Bearer {HF_TOKEN}"} # required if space is private
)

# List of Hugging Face datasets to process
dataset_list = [
    "atlasia/Youtube-Commons-Morocco-Darija-35h",
]

settings = rg.Settings(
    guidelines=ANNOTATION_GUIDELINES,
    fields=[
        rg.TextField(name="audio", use_markdown=True, required=True),
        rg.TextField(name="original_transcription", required=True),
    ],
    questions=[
        rg.RatingQuestion(
            name="transcription_quality",
            title="How good is the transcription? \n(0: Incorrect, 1: Partially Correct, 2: Totally Correct)",
            values=[0, 1, 2],
            required=True
        ),
        rg.TextQuestion(
            name="corrected_transcription",
            title="Provide the correct transcription if needed (only required if rating is 0 or 1)",
            required=False
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
            required=True
        ),
        rg.LabelQuestion(
            name="intent",
            title="Utterance Intent",
            labels=["Question", "Informative/Statement", "Command", "Other"],
            required=True
        ),
        rg.LabelQuestion(
            name="background_noise",
            title="Background Noise Level",
            labels=["None", "Low", "Moderate", "High"],
            required=True
        ),
        rg.LabelQuestion(
            name="speaker_clarity",
            title="Speaker Clarity",
            labels=["Clear", "Somewhat Clear", "Muffled/Unclear"],
            required=True
        ),
        rg.LabelQuestion(
            name="speaker_count",
            title="Speaker Count",
            labels=["Single Speaker", "Multiple Speakers"],
            required=True
        ),
    ],
)

def prepare_audio_for_argilla(example):
    """
    Prepares audio data for logging to Argilla by writing the audio to a file,
    converting it to an HTML snippet using `audio_to_html`, and returning the processed record.
    """
    record_id = example["id"]
    audio_info = example["audio"]
    # Use a unique filename for each audio record
    file_name = f"audio_{record_id}.wav"
    audio_array = audio_info["array"]
    sample_rate = audio_info["sampling_rate"]
    
    # Write the audio data to a file
    sf.write(file_name, audio_array, sample_rate)
    
    # Generate an HTML snippet to embed the audio player
    audio_html = audio_to_html(
        file_name,
        width="300px",
        height="300px",
        autoplay=True,
        loop=True
    )
  
    return {
        "id": record_id,
        "audio": audio_html,
        "original_transcription": example["transcription"],
        "corrected_transcription": example["transcription"]  # default value pre filled
    }

if __name__ == "__main__":
    for hf_dataset_name in dataset_list:
        argilla_dataset_name = hf_dataset_name.split("/")[-1]
        
        hf_dataset = load_dataset(
            hf_dataset_name,
            split="train[:10]",
            token=HF_TOKEN
        )
        
        hf_dataset = hf_dataset.map(lambda example, idx: {"id": idx}, with_indices=True)
        
        dataset = rg.Dataset(
            name=f"{argilla_dataset_name}",
            settings=settings,
            client=client,
        )
        dataset.create()
        
        processed_records = []
        for example in tqdm(hf_dataset, total=len(hf_dataset)):
            processed_record = prepare_audio_for_argilla(example)
            processed_records.append(processed_record)
        
        dataset.records.log(
            records=processed_records,
            mapping={
                "audio": "audio",
                "original_transcription": "original_transcription",
                "corrected_transcription": "corrected_transcription"
            }
        )
        
        print(f"Successfully pushed {len(processed_records)} records to Argilla for dataset: {argilla_dataset_name}")
    
    print("All specified datasets have been pushed to Argilla with the defined settings.")


