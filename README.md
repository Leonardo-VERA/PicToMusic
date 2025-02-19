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

## Phases de Développement
- **Implémentation Grossière**: 
  - Développer une version initiale de la solution avec les fonctionnalités de base.
  - Tester les flux de travail principaux pour s'assurer que chaque composant fonctionne ensemble.

- **Raffinement et Optimisation**: 
  - Améliorer les algorithmes de traitement d'image et de reconnaissance musicale.
  - Optimiser les performances pour réduire la latence et améliorer la précision.
  - Effectuer des tests utilisateurs pour recueillir des retours et ajuster les fonctionnalités.

- **Portage sur Mobile**: 
  - Explorer les options pour porter l'application sur des plateformes mobiles en privilégiant les solutions Python :
    - **Kivy**: Framework Python pour le développement d'applications multiplateformes, y compris Android et iOS.
    - **BeeWare**: Outils pour créer des applications natives en Python pour différentes plateformes, y compris mobile.
    - **PyQt/PySide**: Pour créer des applications avec une interface graphique qui peuvent être adaptées pour mobile, bien que cela nécessite des ajustements supplémentaires.
  - Adapter les fonctionnalités pour une utilisation mobile, en tenant compte des limitations de performance et d'interface.