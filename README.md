# cnn-tesis
# 🎨 Little Picasso - Red Neuronal Convolucional para la Clasificación de Dibujos Infantiles hechos a mano

Proyecto de Tesis red neuronal convolucional educativa que clasifica idetifica dibujos infantiles de animales y genera explicaciones pedagógicas para niños de preescolar.

## 🚀 Tecnologías Clave
| Área       | Tecnologías                                                                 |
|------------|----------------------------------------------------------------------------|
| **Backend** | FastAPI, TensorFlow/Keras, MobileNetV2, Gemini API, ElevenLabs TTS        |
| **Frontend**| Angular, TypeScript, HTML5/CSS3                                           |
| **ML**      | Transfer Learning, Data Augmentation, CNN                                  |
| **DevOps**  | GitHub, Git                                                               |

## ✨ Características Principales
- **Clasificación en 10 clases**: Oso, abeja, mariposa, vaca, perro, mariquita, león, serpiente, estrella de mar, tortuga
- **Respuestas educativas**:
  - Descomposición fonética letra por letra
  - Datos curiosos adaptados a niños
  - Generación de voz (TTS)
- **Validación de confianza**: Filtra predicciones con <35% de confianza

## 🛠️ Instalación
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.api.main:app --reload

# Frontend
cd frontend
npm install
ng serve
