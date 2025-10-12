from roboflow import Roboflow
rf = Roboflow(api_key="7fRjirwWHBv8dx0QpvmT")
project = rf.workspace("yoloconteocartas").project("yolo_conteo_cartas-d2evd")
version = project.version(3)
dataset = version.download("yolov11")
                