import cv2

class PosterType:
    IMAGE = 0
    VIDEO = 1
    THREED = 2  

class Posters:
    def __init__(self):
        self.__paths__ = []
        self.__meta__ = []
        self.__imgExt__ = ('.jpg', '.JPEG', '.jpeg', '.png', '.tiff', '.bmp', '.PNG')
        self.__vidExt__ = ('.mp4', '.avi')

    def add(self, path, *kwargs):
        _meta = (kwargs)
        self.__paths__.append(path)
        self.__meta__.append(_meta)
    
    def init(self, index):
        if self.__paths__[index].endswith(self.__imgExt__):
            print("Image detected!")
            _ret = cv2.imread(self.__paths__[index])
        elif self.__paths__[index].endswith(self.__vidExt__):
            print("Video detected!")
            _ret = cv2.VideoCapture(self.__paths__[index])
        else:
            raise Exception("Path not supported! %s"%(self.__paths__[index]))
        return _ret

POSTERS = Posters()
POSTERS.add('./static/poster1.jpg', ('image', 'cat') )
POSTERS.add('./videos/vibing_cat_trim.mp4', ('image', 'cat'))