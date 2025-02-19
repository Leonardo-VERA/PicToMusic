# PicToMusic

## Roadmap du Projet

### 1. Capture d'Image et Prétraitement
- **Technologie**: OpenCV
- **Fonctionnalités**:
  - Capturer des images depuis une webcam.
  - Prétraiter l'image (désinclinaison, amélioration du contraste, seuillage).
  - Détecter les contours de la partition.

### 2. Prétraitement
- Découper la capture en rectangles et traiter chaque rectangle individuellement.

### 3. Reconnaissance des Notes et du Rythme (OMR - Optical Music Recognition)
- **Technologies**:
  - **Audiveris**: Logiciel open-source spécialisé dans l'OMR pour convertir les images de partitions en fichiers MusicXML ou MIDI.
  - **Alternatives**: 
    - Utiliser un modèle de deep learning personnalisé avec TensorFlow/Keras, entraîné sur des datasets de partitions (ex: MuseScore).
    - **Object Detection**: Utiliser YOLO ou Detectron2 pour détecter les symboles musicaux (clés de sol, notes, silences) dans une approche DIY.
    - **Dataset**: DeepSheet (images annotées de partitions).

### 4. Traitement des Données Musicales
- **Technologie**: Music21
- **Fonctionnalités**:
  - Analyser les fichiers MusicXML/MIDI.
  - Extraire les notes, durées, tempo, etc.

### 5. Synthèse Sonore en Temps Réel
- **Technologies**:
  - **FluidSynth + SoundFonts**: Jouer les notes en direct avec un instrument réaliste (piano, guitare, etc.).
  - **RtMidi**: Bibliothèque C++ avec bindings Python pour une gestion en temps réel du MIDI, assurant une latence ultra-faible.