from roboflow import Roboflow
rf = Roboflow(api_key="7fRjirwWHBv8dx0QpvmT")
project = rf.workspace("yoloconteocartas").project("sequence_tokens-3wuzh")
version = project.version(2)
dataset = version.download("yolov11")
                