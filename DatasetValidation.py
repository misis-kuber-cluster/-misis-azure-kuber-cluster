import os
import cv2
import random
import seaborn as sns
from glob import glob
from matplotlib import pyplot as plt
import kaggle


# Drop kaggle.json file into the C:/Users/**user**/.kaggle folder.
# If it doesn't exist create that.
def download_dataset():
    kaggle.api.authenticate()
    data_directory = "./data"
    # can be long proccess
    print("Downloading...")
    kaggle.api.dataset_download_files("ikarus777/best-artworks-of-all-time", path=data_directory, unzip=True)
    print("Dataset downloaded!")


def print_files_from_directory(directory):
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def is_all_directories_exist(artists_top, images_dir):
    artists_top_name = artists_top['name'].str.replace(' ', '_').values

    for name in artists_top_name:
        if os.path.exists(os.path.join(images_dir, name)):
            print("Found -->", os.path.join(images_dir, name))
        else:
            print("Did not find -->", os.path.join(images_dir, name))


# example: plotImages("Vincent van Gogh", "data/images/images/Vincent_van_Gogh/**")
def PlotImages(artist, directory):
    print(artist)
    multipleImages = glob(directory)
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.subplots_adjust(wspace=0, hspace=0)
    i_ = 0
    for l in multipleImages[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (128, 128))
        plt.subplot(5, 5, i_ + 1)  # .set_title("DirectorVosem'")
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
        plt.axis('off')
        i_ += 1
    plt.imshow()


def show_painting_count_by_artists(artists):
    figsize = (15, 7)
    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    params = {'figure.figsize': figsize,
              'axes.labelsize': labelsize,
              'axes.titlesize': titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)

    col1 = "name"
    col2 = "paintings"

    sns.barplot(x=col1, y=col2, data=artists)
    plt.title("Painting Count by Artist")
    plt.xlabel("Artist")
    plt.ylabel("Painting Count")
    plt.xticks(rotation=90)
    plt.plot()
    plt.show()


def print_random_paintings(image_counts, artists_top_name, images_dir):
    fig, axes = plt.subplots(1, image_counts, figsize=(12, 5))

    for i in range(image_counts):
        random_artist = random.choice(artists_top_name)
        random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
        random_image_file = os.path.join(images_dir, random_artist, random_image)
        image = plt.imread(random_image_file)
        axes[i].imshow(image)
        axes[i].set_title("Artist: " + random_artist.replace('_', ' '))
        axes[i].axis('off')
    plt.show()
