from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import logging
from google import genai
from elevenlabs.client import ElevenLabs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = load_model("../../app/api/modelo_transfer_cnn(shuffle=true, epocas=100, clases=4).h5")
clases = {'oso': 0, 'abeja': 1, 'mariposa': 2, 'vaca': 3, 'perro': 4, 'mariquita': 5, 'leon': 6, 'serpiente': 7, 'estrella de mar': 8, 'tortuga': 9}
clases_invertidas = dict((v, k) for k, v in clases.items())

gemini_client = genai.Client(api_key="AIzaSyBcUtFCMVnEh15G79SM20VqbXqqo18_PVU")

USE_ELEVENLABS = True  # Cambiar a True cuando quieras activar TTS
eleven_client = None

if USE_ELEVENLABS:
    eleven_client = ElevenLabs(api_key="sk_a12f5e5b4d6b0ac8fe0c8a2cdbce321bae58536252c538d8")

def generar_audio(texto: str):
    if not USE_ELEVENLABS or eleven_client is None:
        return None
    
    try:
        audio = eleven_client.text_to_speech.convert(
            voice_id="ajOR9IDAaubDK5qtLUqQ",
            text=texto,
            model_id="eleven_multilingual_v2",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.2,
                "speaker_boost": True
            }
        )
        audio_bytes = b"".join(audio)
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Error generando audio: {str(e)}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

       
        logger.info(f"Imagen procesada - Shape: {img_array.shape}, Rango: {img_array.min()} - {img_array.max()}")

       
        prediccion = modelo.predict(img_array)
        indice = np.argmax(prediccion)
        clase_predicha = clases_invertidas[indice].capitalize()
        confianza = float(prediccion[0][indice])
        
        logger.info(f"Predicción: {clase_predicha} (Confianza: {confianza:.2%})")
        logger.debug(f"Todas las predicciones: {dict(zip(clases.keys(), prediccion[0]))}")

  
        prompt = f"""
Eres una IA llamada Little Picasso para niños de preescolar. Tus respuestas serán escuchadas por niños mediante voz, así que habla de forma divertida, clara y comprensiva para ellos (niños de 4 a 6 años). No uses símbolos, pero sí puedes usar exclamaciones y preguntas para hacerlas llamativas.

Has recibido el dibujo de un niño que representa a un: {clase_predicha}.

Instrucciones ESPECÍFICAS:

1. Felicita al niño por su dibujo y dile qué animal es.
2. *DESCOMPOSICIÓN LETRA POR LETRA* (NO por sílabas):
   - Debes descomponer la palabra {clase_predicha.upper()} LETRA POR LETRA
   - Cada letra debe presentarse COMO SE PRONUNCIA (no como se escribe)
   - Formato OBLIGATORIO: "[letra pronunciada]...de [palabra ejemplo]..."
   - Ejemplos CORRECTOS:
     * "OSO" -> "O...de Ola..., Ese...de Sapo..., O...de Oso..." (NUNCA "O...de Ola..., So...de Sol...")
     * "ABEJA" -> "A...de Avión..., Be...de Beso..., E...de Elefante..., Jota...de Jirafa..., A...de Araña..."
     * "MARIPOSA" -> "Eme...de Mamá..., A...de Avión..., Ere...de Ratón..., I...de Igual..., Pe...de Papá..., O...de Oso..., Ese...de Sapo..., A...de Árbol..."
   - Reglas estrictas:
     * Nunca separes por sílabas, siempre por letras individuales
     * Usa la pronunciación fonética de cada letra (ej. "Be" no "B", "Ese" no "S")
     * Incluye todas las letras, incluso si son mudas (como la 'h')
3. Datos curiosos:
   - Proporciona uno o dos datos sobre el animal
   - Que sean sorprendentes, divertidos y fáciles de entender

Responde todo seguido y como si le estuvieras hablando directamente al niño. ¡Usa mucho entusiasmo!
"""
        try:
            respuesta = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            texto_respuesta = respuesta.text
            logger.info("Respuesta de Gemini generada exitosamente")
        except Exception as e:
            logger.error(f"Error con Gemini: {str(e)}")
            texto_respuesta = f"¡Buen trabajo dibujando un {clase_predicha}! ¿Quieres aprender más sobre este animal?"

     
        audio_base64 = generar_audio(texto_respuesta) if USE_ELEVENLABS else None
        return JSONResponse(content={
            "animal": clase_predicha,
            "mensaje": texto_respuesta,
            "audio": audio_base64,
            "metadata": {
                "confianza": confianza,
                "tts_activado": USE_ELEVENLABS,
                "clases_disponibles": list(clases.keys())
            }
        })

    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/debug_predict")
async def debug_predict(file: UploadFile = File(...)):
    """Endpoint para diagnóstico técnico (opcional)"""
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        
        return JSONResponse(content={
            "metadata": {
                "formato": img.format,
                "modo": img.mode,
                "tamaño_original": img.size,
                "tamaño_procesado": (256, 256)
            },
            "notas": "Esta es una respuesta de diagnóstico para verificar el preprocesamiento"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )