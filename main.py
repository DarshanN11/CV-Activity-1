
import cv2
import numpy as np

# ----------------------------
# Load image
# ----------------------------
img = cv2.imread('input.jpg')  # change path if needed
if img is None:
    raise FileNotFoundError("Image not found. Please place 'input.jpg' in this folder.")

# 1. Display image in color and grayscale
cv2.imshow('Color Image', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)

# 2. Save image in different format (JPG -> PNG)
cv2.imwrite('output.png', img)

# 3. Resize image to fixed width and height
resized = cv2.resize(img, (300, 300))
cv2.imshow('Resized Image', resized)

# 4. Flip image horizontally and vertically
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)
cv2.imshow('Horizontal Flip', flip_h)
cv2.imshow('Vertical Flip', flip_v)

# 5. Crop ROI
h, w = img.shape[:2]
roi = img[50:250, 50:250]
cv2.imshow('Cropped ROI', roi)

# 6. Draw rectangle, circle, and line
shape_img = img.copy()
cv2.rectangle(shape_img, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.circle(shape_img, (300, 150), 50, (255, 0, 0), 2)
cv2.line(shape_img, (0, 0), (w, h), (0, 0, 255), 2)
cv2.imshow('Shapes', shape_img)

# 7. Put custom text (your name)
text_img = img.copy()
cv2.putText(text_img, 'Darshan Gowda', (50, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('Text Image', text_img)

# 8. Convert to HSV and extract red color
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2
red_only = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('Red Regions', red_only)

# 9. Split BGR channels
b, g, r = cv2.split(img)
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)

# 10. Merge channels back
merged = cv2.merge([b, g, r])
cv2.imshow('Merged Image', merged)

# 11. Adjust brightness
bright = cv2.convertScaleAbs(img, alpha=1, beta=50)
cv2.imshow('Bright Image', bright)

# 12. Adjust contrast
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
cv2.imshow('High Contrast Image', contrast)

# 13. Gaussian blur
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Gaussian Blur', gaussian)

# 14. Median blur
median = cv2.medianBlur(img, 5)
cv2.imshow('Median Blur', median)

# 15. Rotate 90, 180, 270 degrees
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot180 = cv2.rotate(img, cv2.ROTATE_180)
rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('Rotate 90', rot90)
cv2.imshow('Rotate 180', rot180)
cv2.imshow('Rotate 270', rot270)

# 16. Rotate arbitrary angle (45°) without cropping
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
nW = int((h * sin) + (w * cos))
nH = int((h * cos) + (w * sin))
M[0, 2] += (nW / 2) - center[0]
M[1, 2] += (nH / 2) - center[1]
rot45 = cv2.warpAffine(img, M, (nW, nH))
cv2.imshow('Rotate 45', rot45)

# 17. Translate image
M_trans = np.float32([[1, 0, 50], [0, 1, 50]])
translated = cv2.warpAffine(img, M_trans, (w, h))
cv2.imshow('Translated Image', translated)

# 18. Binary image using threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Image', binary)

# 19. Canny edge detection
edges = cv2.Canny(gray, 100, 200)
cv2.imshow('Edges', edges)

# 20. Overlay smaller image (logo) on corner
logo = cv2.imread('logo.png')  # provide a small logo image
if logo is not None:
    lh, lw = logo.shape[:2]
    img_overlay = img.copy()
    img_overlay[0:lh, 0:lw] = logo
    cv2.imshow('Overlay Image', img_overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
