from roboflow import Roboflow
rf = Roboflow(api_key="XMzecNvykRnOA4HvsARi")
project = rf.workspace("harshitas-workspace-hnzai").project("pune_traffic_fix")
version = project.version(2)
dataset = version.download("yolov8")
                