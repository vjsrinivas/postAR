# postAR

[![Devpost Badge Time](https://badges.devpost-shields.com/get-badge?name=postAR&id=postar-ajtm7s&type=basic&style=plastic)](https://devpost.com/software/postar-ajtm7s)

### Won "Best Solo Hacker 2nd Place" and "Most Useless"! 😄

---

**Descrption:**

A VolHack 2021 Project - Projecting a poster or digital item onto a green screen in realtime.

![The projection feed working on the compute machine](./assets/overview.gif)

**[Video Link](https://www.youtube.com/watch?v=qtkfbua-O0Y)**

## Requirements

### Hardware:
- Google Cardboard Headset
- Luxonis OAK-D
- Host Computer (Windows or Linux-based)

### Sofware:
- [DepthAI API]()
- numPy
- OpenCV2 (2+)
- imutils
- tKinter

## How It Works

![Pipeline of the projection system](./assets/pipeline.png)

I utilized various image processing techniques to each a reliable projection method. The first component would be the HSV-based segmentation method, which is a fairly simple method of segmentation. The next components are simply to reduce the complexity of the features from the segmentation masks and a method to find a rectangle from the reduced mask. From then, it is just a simple distortion and split screen filter to create a VR-like view. Since all of this is happening on the laptop, we would want to stream it to the display (a phone) via MJPEG streaming, which just uses a HTML webpage.
