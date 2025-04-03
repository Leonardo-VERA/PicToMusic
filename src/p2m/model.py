from ultralytics import YOLO

def train(data_path: str, model_path: str = "yolo11n.pt", **kwargs):
    """
    Train a YOLO model on a custom dataset.

    Args:
        data_path (str): Path to dataset configuration file (e.g. 'data/dataset.yaml')
        model_path (str, optional): Path to initial model weights.
        **kwargs: Additional training arguments including:
            epochs (int, optional): Number of training epochs.
            batch (int): Batch size (-1 for autobatch)
            imgsz (int): Input image size
            device (int|str): Device to run on (e.g. cuda device=0 or device=0,1,2,3 or device=cpu)
            workers (int): Number of worker threads for data loading (per RANK if DDP)
            optimizer (str): Optimizer to use (e.g. SGD, Adam, AdamW)
            patience (int): Epochs to wait for no observable improvement for early stopping

    For more training options, see:
    https://docs.ultralytics.com/fr/modes/train/#train-settings

    Returns:
        YOLO: Trained YOLO model instance
    """
    model = YOLO(model_path) 
    model.train(data=data_path, **kwargs)
    return model


def predict(model_path: str, image_path: str, save : bool = False, save_path: str = None):
    model = YOLO(model_path)
    return model.predict(image_path, save=save, project=save_path)

if __name__ == "__main__":
    train(data_path="data/YOLOv3/dataset.yaml", 
          model_path="models/yolo11n.pt", 
          epochs=100,
          batch=-1,
          workers=8,
          device=0,
          patience=10,
          optimizer='AdamW',
          imgsz=320
          )