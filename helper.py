from transformers import pipeline
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor, pipeline
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import gradio as gr 
import requests
import numpy as np
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

def start_conversation():
    whisper_model_path= "./models/automatic-speech-recognition"
    processor = AutoProcessor.from_pretrained(whisper_model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, use_safetensors=True)
    output= ""
    torch_dtype = torch.float32

    whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype
    )

    demo = gr.Blocks()

    def transcribe_speech(file_path):
        if file_path is None:
            gr.Warning("No audio found please try again")
            return ""
        output = whisper_pipe(file_path)
        creating_conversation_with_blenderbot(output["text"])
        return output["text"]
        
    mic_transcribe = gr.Interface(
    fn =transcribe_speech,
    inputs = gr.Audio(sources="microphone",
                      type="filepath"),
    outputs = gr.Textbox(label="Transcription_results",
                         lines=3),
    allow_flagging="never")
    
    def launch_demo():
        with demo:
            gr.TabbedInterface(
                [mic_transcribe],
                ["Transcribe Microphone"]
            )
            demo.launch(debug=True, share=True)
    launch_demo()

def creating_conversation_with_blenderbot(transcribed_text, conversation_history=[]):
    from transformers import Conversation, BlenderbotTokenizer, BlenderbotForConditionalGeneration
    blender_model_path = "./models/facebook"
    
    tokenizer = BlenderbotTokenizer.from_pretrained(blender_model_path)
    model = BlenderbotForConditionalGeneration.from_pretrained(blender_model_path)

    if conversation_history:
        transcribed_text = '\n'.join(conversation_history + [transcribed_text])

    inputs = tokenizer([transcribed_text], return_tensors="pt", max_length=128, truncation=True)

    reply_ids = model.generate(**inputs)
    
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    conversation_history.append(transcribed_text)
    conversation_history.append(response)

    produce_reply_sound(response)
    return conversation_history
def produce_reply_sound(text):
    import pyttsx3
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate- 50)
    engine.say(text)
    engine.runAndWait()