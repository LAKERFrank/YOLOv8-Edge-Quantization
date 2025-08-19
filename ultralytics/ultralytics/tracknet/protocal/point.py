from typing import Protocol
import numpy as np

class PointProtocol(Protocol):
    fid: int
    timestamp: float
    visibility: int
    x: float
    y: float
    z: float
    event: int
    speed: float
    color: str
    cam_idx: int

    def toJson(self) -> dict: ...
    def build_string(self) -> str: ...
    def setX(self, x: float) -> None: ...
    def setY(self, y: float) -> None: ...
    def setZ(self, z: float) -> None: ...
    def toXY(self) -> np.ndarray: ...
    def toXYT(self) -> np.ndarray: ...
    def toXYZ(self) -> np.ndarray: ...
    def toXYZT(self) -> np.ndarray: ...
    def __str__(self) -> str: ...
    def __lt__(self, other: object) -> bool: ...

class Point(PointProtocol):
    def __init__(self, fid=-1, timestamp=-1, visibility=0, x=0, y=0, z=0, event=0, speed=0.0, color='white'):
        self.fid = int(float(fid))
        self.timestamp = float(timestamp)
        self.visibility = int(float(visibility))
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.event = int(float(event))
        self.speed = float(speed)
        self.color = color
        self.cam_idx = -1

    @staticmethod
    def fromJson(data:dict):
        return Point(data["id"], data["timestamp"], data["visibility"], data["pos"]["x"], data["pos"]["y"], data["pos"]["z"], data["event"], data["speed"], data["color"])

    def toJson(self):
        """Json serializable for MQTT

        Returns:
            dict: data
        """
        return {
            "id": self.fid,
            "timestamp": self.timestamp,
            "visibility": self.visibility,
            "pos": {
                "x":self.x,
                "y":self.y,
                "z":self.z
            },
            "event": self.event,
            "speed": self.speed,
            "color": self.color
        }

    def build_string(self):
        payload = {"id": self.fid, "timestamp": self.timestamp, "visibility": self.visibility,
            "pos": {"x":self.x, "y":self.y, "z":self.z}, "event": self.event, "speed": self.speed, "color": self.color}
        return json.dumps(payload)

    def setX(self, x):
        self.x = float(x)

    def setY(self, y):
        self.y = float(y)

    def setZ(self, z):
        self.z = float(z)

    def __str__(self):
        s = (f"\nPoint Fid: {self.fid}\n"
             f"Timestamp: {self.timestamp:>.3f}\n"
             f"Vis: {self.visibility}\n"
             f"({self.x:.2f},{self.y:.2f},{self.z:.2f})\n"
             f"Event: {self.event}\n"
             f"Speed: {self.speed:.0f}\n"
             f"Camera Idx: {self.cam_idx}\n")
        return s

    def toXY(self):
        return np.array([self.x, self.y])

    def toXYT(self):
        return np.array([self.x, self.y, self.timestamp])

    def toXYZ(self):
        return np.array([self.x, self.y, self.z])

    def toXYZT(self):
        return np.array([self.x, self.y, self.z, self.timestamp])

    def __lt__(self, other):
        # sorted by timestamp
        return self.timestamp < other.timestamp
