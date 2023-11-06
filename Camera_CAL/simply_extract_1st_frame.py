import cv2
a=11;
paths=["Camera_CAL//1.mp4","Camera_CAL//2.mp4","Camera_CAL//3.mp4","Camera_CAL//4.mp4","Camera_CAL//5.mp4","Camera_CAL//6.mp4"];
for i in paths:
    video=cv2.VideoCapture(i);
    _,row=video.read();
    print(row.shape);
    name=str(a)+".jpg";
    cv2.imwrite(name,row);
    a+=1;