import matplotlib.pylab as plt
import cv2
import numpy as np

def roi(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

#image = cv2.imread(r'C:\Users\hp\Desktop\lane1.jpeg')  For image purpose
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    roi_vertices = [
        (0, height),
        (width/1.8, height/1.8),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image,(5,5),0)
    canny_image = cv2.Canny(blur_image, 100, 120)
    cropped_image = roi(canny_image,
                    np.array([roi_vertices], np.int32),)

    lines = cv2.HoughLinesP(cropped_image,
                            rho=4,
                            theta=np.pi/180,
                            threshold=100,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=50)

    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('road.mp4')

while(cap.isOpened()):
    ret,frame = cap.read()
    frame = process(frame)
    cv2.imshow('Lane', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





    
