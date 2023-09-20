import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import ipdb

########## Section 2.2 <Loading and displaying images>#######################
# Read the image
imgObj = img.imread("testImage.png")
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the raw image
plt.imshow(imgObj)
plt.show()

# Change image to grayscale
grayImgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2GRAY)
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the grayscaled image
plt.imshow(grayImgObj, cmap = 'gray')
plt.show()

# specify dimensions for image zoom
left, right, top, bottom = 100, 300, 250, 510

# Zoom into the image (crop operation)
zoomedImage = imgObj[top:bottom, left:right]
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the zoomed in image
plt.imshow(zoomedImage)
plt.show()
##############################################
'''
########## Section 2.3 <Resizing images>#######################
# Read the image
imgObj = img.imread("poseSit.png")
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the raw image
plt.imshow(imgObj)
plt.show()

# Specify new dimensions (downsample height and width by factor of 10)
newHeight = imgObj.shape[0] // 10
newWidth = imgObj.shape[1] // 10

# Downsample image by resizing
downsampledImgObj = cv2.resize(imgObj, (newWidth, newHeight))
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the downsampled image
plt.imshow(downsampledImgObj)
plt.show()

# Specify new dimensions (upsample height and width by factor of 10)
newHeight = downsampledImgObj.shape[0] * 10 + 1
newWidth = downsampledImgObj.shape[1] * 10 + 2

# Upsample image by resizing (Nearest neighbor interpolation)
newHeight = imgObj.shape[0] // 10
newWidth = imgObj.shape[1] // 10
downsampledImgObj = cv2.resize(imgObj, (newWidth, newHeight))

newHeight = downsampledImgObj.shape[0] * 10 + 1
newWidth = downsampledImgObj.shape[1] * 10 + 2
nnUpsampledImgObj = cv2.resize(downsampledImgObj, (newWidth, newHeight), interpolation = cv2.INTER_NEAREST)
bicubicUpsampledImgObj = cv2.resize(downsampledImgObj, (newWidth, newHeight), interpolation = cv2.INTER_CUBIC)
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the upsampled image
plt.imshow(nnUpsampledImgObj)
plt.show()

# Upsample image by resizing (Bicubic interpolation)
bicubicUpsampledImgObj = cv2.resize(downsampledImgObj, (newWidth, newHeight), interpolation = cv2.INTER_CUBIC)
plt.xticks([])

# disabling yticks by setting yticks to an empty list
plt.yticks([])
# Display the upsampled image
plt.imshow(bicubicUpsampledImgObj)
plt.show()

nnDiff = cv2.absdiff(imgObj, nnUpsampledImgObj)
bicubicDiff = cv2.absdiff(imgObj, bicubicUpsampledImgObj)

# Display the upsampled image
cv2.imshow("nndiff", nnDiff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the upsampled image
cv2.imshow("bicubicDiff", bicubicDiff)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(np.sum(nnDiff))
print(np.sum(bicubicDiff))

##############################################

imgObj = img.imread("testImage.png")

grayImgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2GRAY)


# Compute the 2D Fourier Transform
fourier_transform = np.fft.fft2(grayImgObj)
fourier_transform_shifted = np.fft.fftshift(fourier_transform)  # Shift zero frequency components to the center

# Calculate magnitude and phase
magnitude = np.abs(fourier_transform_shifted)
phase = np.angle(fourier_transform_shifted)
'''
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(grayImgObj, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(magnitude), cmap='gray')
plt.title('Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(phase, cmap='gray')
plt.title('Phase')
plt.axis('off')

plt.tight_layout()
newHeight = imgObj.shape[0] // 10
newWidth = imgObj.shape[1] // 10
downsampledImgObj = cv2.resize(imgObj, (newWidth, newHeight))   # Downsample 10x

newHeight = downsampledImgObj.shape[0] * 10 + 1
newWidth = downsampledImgObj.shape[1] * 10 + 2
nnUpsampledImgObj = cv2.resize(downsampledImgObj, (newWidth, newHeight), interpolation = cv2.INTER_NEAREST)     #Upsample 10x
bicubicUpsampledImgObj = cv2.resize(downsampledImgObj, (newWidth, newHeight), interpolation = cv2.INTER_CUBIC)      #Upsample 10x



plt.show()
'''
f_transform = np.fft.fft2(grayImgObj)
f_transform_shifted = np.fft.fftshift(f_transform)

rows, cols = grayImgObj.shape
center_row, center_col = rows // 2, cols // 2

cutoff_low = 20  # Cutoff frequency for low-pass filter
cutoff_high = 100  # Cutoff frequency for high-pass filter
radius_bandpass_low = 20  # Lower cutoff frequency for bandpass filter
radius_bandpass_high = 100  # Upper cutoff frequency for bandpass filter


def butterworth_low_pass(rows, cols, center_row, center_col, cutoff, n=1):
    x = np.arange(cols) - center_col
    y = np.arange(rows) - center_row
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx ** 2 + yy ** 2)

    # Butterworth low-pass filter formula
    filter = 1 / (1 + (distance / cutoff) ** (2 * n))
    return filter


def butterworth_high_pass(rows, cols, center_row, center_col, cutoff, n=1):
    return 1 - butterworth_low_pass(rows, cols, center_row, center_col, cutoff, n)


def butterworth_bandpass(rows, cols, center_row, center_col, radius_low, radius_high, n=1):
    low_pass = butterworth_low_pass(rows, cols, center_row, center_col, radius_low, n)
    high_pass = butterworth_high_pass(rows, cols, center_row, center_col, radius_high, n)
    return low_pass * high_pass


n_value = 2  
low_pass_filter = butterworth_low_pass(rows, cols, center_row, center_col, cutoff_low, n_value)
high_pass_filter = butterworth_high_pass(rows, cols, center_row, center_col, cutoff_high, n_value)
bandpass_filter = butterworth_bandpass(rows, cols, center_row, center_col, radius_bandpass_low, radius_bandpass_high,
                                       n_value)

low_pass_result = f_transform_shifted * low_pass_filter
high_pass_result = f_transform_shifted * high_pass_filter
bandpass_result = f_transform_shifted * bandpass_filter

low_pass_image = np.fft.ifftshift(low_pass_result)
low_pass_image = np.fft.ifft2(low_pass_image).real

high_pass_image = np.fft.ifftshift(high_pass_result)
high_pass_image = np.fft.ifft2(high_pass_image).real

bandpass_image = np.fft.ifftshift(bandpass_result)
bandpass_image = np.fft.ifft2(bandpass_image).real

fourier_transform = np.fft.fft2(grayImgObj)
fourier_transform_shifted = np.fft.fftshift(fourier_transform)  

magnitude0 = np.abs(fourier_transform_shifted)
phase = np.angle(fourier_transform_shifted)

fourier_transform = np.fft.fft2(low_pass_image)
fourier_transform_shifted = np.fft.fftshift(fourier_transform)  

magnitude1 = np.abs(fourier_transform_shifted)
phase = np.angle(fourier_transform_shifted)

fourier_transform = np.fft.fft2(high_pass_image)
fourier_transform_shifted = np.fft.fftshift(fourier_transform) 

magnitude2 = np.abs(fourier_transform_shifted)
phase = np.angle(fourier_transform_shifted)

fourier_transform = np.fft.fft2(bandpass_image)
fourier_transform_shifted = np.fft.fftshift(fourier_transform)  

# Calculate magnitude and phase
magnitude3 = np.abs(fourier_transform_shifted)
phase = np.angle(fourier_transform_shifted)

'''
plt.figure(figsize=(12, 8))

plt.subplot(1, 4, 1)
plt.imshow(grayImgObj, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(low_pass_image, cmap='gray')
plt.title('Low-Pass Filtered Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(high_pass_image, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(bandpass_image, cmap='gray')
plt.title('Bandpass Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()
'''
'''
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(np.log1p(magnitude0), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.log1p(magnitude1), cmap='gray')
plt.title('Low-Pass Filter')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.log1p(magnitude2), cmap='gray')
plt.title('High-Pass Filter')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(np.log1p(magnitude3), cmap='gray')
plt.title('Band-Pass Filter')
plt.axis('off')

plt.tight_layout()
plt.show()
'''
'''
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(grayImgObj, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(low_pass_image, cmap='gray')
plt.title('Low-Pass Filter')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(high_pass_image, cmap='gray')
plt.title('High-Pass Filter')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(high_pass_image, cmap='gray')
plt.title('Band-Pass Filter')
plt.axis('off')
plt.tight_layout()
plt.show()
'''


'''
fft_image1 = np.fft.fft2(cv2.imread())
fft_image2 = np.fft.fft2(imgObj2)

fft_image1_shifted = np.fft.fftshift(fft_image1)
fft_image2_shifted = np.fft.fftshift(fft_image2)

magnitude_image1 = np.abs(fft_image1_shifted)
magnitude_image2 = np.abs(fft_image2_shifted)

phase_image1 = np.angle(fft_image1_shifted)
phase_image2 = np.angle(fft_image2_shifted)

plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(np.log1p(magnitude_image1), cmap='gray')
plt.title('Magnitude Image 1')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(phase_image1, cmap='gray')
plt.title('Phase Image 1')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.log1p(magnitude_image2), cmap='gray')
plt.title('Magnitude Image 2')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(phase_image2, cmap='gray')
plt.title('Phase Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()
'''

imgObj1 = cv2.cvtColor(img.imread("1.png"), cv2.COLOR_BGR2GRAY)
imgObj2 = cv2.cvtColor(img.imread("2.png"), cv2.COLOR_BGR2GRAY)

fft_image1 = np.fft.fft2(imgObj1)
fft_image2 = np.fft.fft2(imgObj2)

magnitude_image1 = np.abs(fft_image1)
magnitude_image2 = np.abs(fft_image2)

phase_image1 = np.angle(fft_image1)
phase_image2 = np.angle(fft_image2)

combined_image1 = np.multiply(magnitude_image1, np.exp(1j * phase_image2))
combined_image2 = np.multiply(magnitude_image2, np.exp(1j * phase_image1))


image_swapped_phase_1 = np.fft.ifft2(combined_image1).real
image_swapped_phase_2 = np.fft.ifft2(combined_image2).real

plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(imgObj1, cmap='gray')
plt.title('Image 1 (Original)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(imgObj2, cmap='gray')
plt.title('Image 2 (Original)')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(image_swapped_phase_1, cmap='gray')
plt.title('Phase Swapped: Image 2-Image 1')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(image_swapped_phase_2, cmap='gray')
plt.title('Phase Swapped: Image 1-Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()

'''


image1 = cv2.cvtColor(img.imread("1.png"), cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(img.imread("2.png"), cv2.COLOR_BGR2GRAY)

fft_image1 = np.fft.fft2(image1)
fft_image2 = np.fft.fft2(image2)

cutoff_low = 100
cutoff_high = 100


def butterworth_low_pass(rows, cols, center_row, center_col, cutoff, n=1):
    x = np.arange(cols) - center_col
    y = np.arange(rows) - center_row
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx ** 2 + yy ** 2)

    filter = 1 / (1 + (distance / cutoff) ** (2 * n))
    return filter


def butterworth_high_pass(rows, cols, center_row, center_col, cutoff, n=1):
    return 1 - butterworth_low_pass(rows, cols, center_row, center_col, cutoff, n)


low_pass_filter = butterworth_low_pass(image1.shape[0], image1.shape[1], image1.shape[0] // 2, image1.shape[1] // 2,
                                       cutoff_low)
high_pass_filter = butterworth_high_pass(image2.shape[0], image2.shape[1], image2.shape[0] // 2, image2.shape[1] // 2,
                                         cutoff_high)

filtered_image1 = fft_image1 * low_pass_filter
filtered_image2 = fft_image2 * high_pass_filter

filtered_image1 = np.fft.ifftshift(filtered_image1)
filtered_image1 = np.fft.ifft2(filtered_image1).real

filtered_image2 = np.fft.ifftshift(filtered_image2)
filtered_image2 = np.fft.ifft2(filtered_image2).real

combined_image = filtered_image1 + filtered_image2

plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1 (Original)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2 (Original)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(filtered_image1, cmap='gray')
plt.title('Image 1 (Low-Pass Filtered)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(filtered_image2, cmap='gray')
plt.title('Image 2 (High-Pass Filtered)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Image')
plt.axis('off')

#plt.tight_layout()
#plt.show()
'''

import cv2
import numpy as np

width, height = 400, 400  
alpha_channel = np.ones((height, width), dtype=np.float32)

for y in range(height):
    alpha_channel[y, :] = (y / height)

image1 = cv2.imread('/Users/hariiyer/Desktop/IMG_3894.jpeg')
image2 = cv2.imread('/Users/hariiyer/Desktop/IMG_3895.jpeg')

alpha_channel_normalized = alpha_channel / np.max(alpha_channel)

result = cv2.addWeighted(image1, 1 - alpha_channel_normalized, image2, alpha_channel_normalized, 0)

cv2.imwrite('output.jpg', result)
cv2.imshow('Blended Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

video_file = '/Users/hariiyer/Desktop/Hari Iyer/EEE 515/HW1_Canvas/livingroom.MOV'

output_file = 'difference_output.mp4'

cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

ret, reference_frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size, isColor=True)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    diff_frame = cv2.absdiff(frame, reference_frame)

    out.write(diff_frame)

cap.release()
out.release()

print(f"Difference video saved as {output_file}")
