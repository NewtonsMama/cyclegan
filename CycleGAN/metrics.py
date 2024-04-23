import os
from skimage import io
from skimage.metrics import structural_similarity as ssim
# Path to the test images folder
test_folder_1 = '/Users/mihit/Downloads/cyclegan 4/CycleGAN/test_images/test/gen_zebra'
test_folder_2 = '/Users/mihit/Downloads/cyclegan 4/CycleGAN/test_images/test/ori_zebra'

# Get a list of image files in the folder
image_files_1 = [os.path.join(test_folder_1, file) for file in os.listdir(test_folder_1) if file.endswith('.png')]
image_files_2 = [os.path.join(test_folder_2, file) for file in os.listdir(test_folder_1) if file.endswith('.png')]

# Compute SSIM for each pair of images
mean_ssim_score = 0
for i in range(len(image_files_1)):
    image1 = io.imread(image_files_1[i])
    image2 = io.imread(image_files_2[i])
    ssim_score = ssim(image1, image2, multichannel=True)
    mean_ssim_score += ssim_score
    
mean_ssim_score = mean_ssim_score/len(image_files_1)
print("CT :",mean_ssim_score)


# Path to the test images folder
test_folder_1 = '/Users/mihit/Downloads/cyclegan 4/CycleGAN/test_images/test/gen_horse'
test_folder_2 = '/Users/mihit/Downloads/cyclegan 4/CycleGAN/test_images/test/ori_horse'

# Get a list of image files in the folder
image_files_1 = [os.path.join(test_folder_1, file) for file in os.listdir(test_folder_1) if file.endswith('.png')]
image_files_2 = [os.path.join(test_folder_2, file) for file in os.listdir(test_folder_1) if file.endswith('.png')]

# Compute SSIM for each pair of images
mean_ssim_score = 0
for i in range(len(image_files_1)):
    image1 = io.imread(image_files_1[i])
    image2 = io.imread(image_files_2[i])
    ssim_score = ssim(image1, image2, multichannel=True)
    mean_ssim_score += ssim_score
    
mean_ssim_score = mean_ssim_score/len(image_files_1)
print("MRI :",mean_ssim_score)

