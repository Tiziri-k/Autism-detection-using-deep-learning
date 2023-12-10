from bing_image_downloader import downloader
import os
import glob


dir = 'Non Autistic'
downloader.download('sad children face', limit = 100, output_dir = dir, adult_filter_off = True)
downloader.download('happy children face', limit = 100, output_dir = dir, adult_filter_off = True)
downloader.download('fear children face', limit = 100, output_dir = dir, adult_filter_off = True)
downloader.download('angry children face', limit = 100, output_dir = dir, adult_filter_off = True)
downloader.download('surprise children face', limit = 100, output_dir = dir, adult_filter_off = True)

dir_1 = '/Non Autistic/happy children face/'
dir_2 = 'Non Autistic/fear children face/'
dir_3 = 'Non Autistic/angry children face/'
dir_4 = 'Non Autistic/sad children face/'
dir_5 = 'Non Autistic/surprise children face/'


for file in os.listdir(dir_1):
    new_file = file.replace("Image","happy")
    os.rename(dir_1 + file, dir_1 + new_file)


for file in os.listdir(dir_2):
    new_file = file.replace("Image","fear")
    os.rename(dir_2 + file, dir_2 + new_file)

for file in os.listdir(dir_3):
    new_file = file.replace("Image","angry")
    os.rename(dir_3 + file, dir_3 + new_file)

for file in os.listdir(dir_4):
    new_file = file.replace("Image","sad")
    os.rename(dir_4 + file, dir_4 + new_file)

for file in os.listdir(dir_5):
    new_file = file.replace("Image","surprise")
    os.rename(dir_5 + file, dir_5 + new_file)


dataset_path = 'Non Autistic/happy children face/'
class_names = 'Happy'


ds_path = os.path.join(dataset_path, class_names[1], '*')
ds_path = glob.glob(ds_path)