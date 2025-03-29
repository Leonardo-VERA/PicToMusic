from ultralytics import YOLO

def train(data_path: str, model_path: str = "yolo11n.pt", epochs: int = 200):
    model = YOLO(model_path) 
    model.train(data=data_path, epochs=epochs)
    return model

def predict(model_path: str, image_path: str, save : bool = False, save_path: str = None):
    model = YOLO(model_path)
    return model.predict(image_path, save=save, project=save_path)

