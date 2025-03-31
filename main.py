from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
from openai import OpenAI
import edge_tts
import io
import os
import base64
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Conversation history storage
conversation_histories = {}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    
    # Initialize conversation history
    conversation_histories[client_id] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    try:
        while True:
            # Receive audio data as bytes
            audio_data = await websocket.receive_bytes()
            logger.info(f"Received audio chunk: {len(audio_data)} bytes")
            
            # Transcribe audio
            transcript = await transcribe_audio(audio_data)
            if not transcript.strip():
                logger.warning("Empty transcript received")
                continue
                
            logger.info(f"Transcript: {transcript}")
            
            # Add to conversation history
            conversation_histories[client_id].append(
                {"role": "user", "content": transcript}
            )
            
            # Generate response
            response_text = await generate_openai_response(
                conversation_histories[client_id]
            )
            logger.info(f"Generated response: {response_text}")
            
            # Add assistant response to history
            conversation_histories[client_id].append(
                {"role": "assistant", "content": response_text}
            )
            
            # Convert to speech
            audio_response = await text_to_speech(response_text)
            
            # Send back audio
            if audio_response:
                await websocket.send_bytes(audio_response)
                logger.info("Sent audio response")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Clean up
        conversation_histories.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected")

async def transcribe_audio(audio_bytes):
    """Transcribe audio using Whisper API"""
    if not client:
        logger.error("OpenAI client not initialized")
        return ""
    
    try:
        # Create in-memory file with webm format
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"
        
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            content_type="audio/webm"
        )
        return transcription.text
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return ""

async def generate_openai_response(conversation_history):
    """Generate response using GPT"""
    if not client:
        logger.error("OpenAI client not initialized")
        return "I'm having trouble connecting to the AI service."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using more affordable model
            messages=conversation_history,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT generation failed: {e}", exc_info=True)
        return "I encountered an error processing your request."

async def text_to_speech(text):
    """Convert text to speech using Edge TTS"""
    if not text.strip():
        return b''
    
    try:
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        output_bytes = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_bytes.write(chunk["data"])

        return output_bytes.getvalue()
    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        return b''

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)