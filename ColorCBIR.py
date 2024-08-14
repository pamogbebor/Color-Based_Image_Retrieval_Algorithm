import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance

# This function extracts the color histogram of images. The parameters change according to the color space.
def calculateHistogram(image, bins=(8, 8, 8), color_space='RGB'):
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 181, 0, 256, 0, 256])
    elif color_space == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    elif color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    else:
        raise ValueError("Unknown method: choose from 'HSV', 'RGB', 'LAB'")
    hist = cv2.normalize(hist, hist).flatten()
    return hist

#This function compares the histograms of the query image and database. The calculations change according to the equation.
def compareHistograms(histA, histB, method='euclidean'):
    if method == 'euclidean':
        return distance.euclidean(histA, histB)
    elif method == 'cosine':
        return distance.cosine(histA, histB)
    elif method == 'chi2':
        return cv2.compareHist(histA.astype('float32'), histB.astype('float32'), cv2.HISTCMP_CHISQR)
    elif method == 'correlation':
        return cv2.compareHist(histA.astype('float32'), histB.astype('float32'), cv2.HISTCMP_CORREL)
    elif method == 'intersection':
        return cv2.compareHist(histA.astype('float32'), histB.astype('float32'), cv2.HISTCMP_INTERSECT)
    elif method == 'bhattacharyya':
        return cv2.compareHist(histA.astype('float32'), histB.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
    else:
        raise ValueError("Unknown method: choose from 'euclidean', 'cosine', 'chi2', 'correlation', 'intersection', or 'bhattacharyya'")


# This functions saves all the color histogram data (RGB & HSV) into a csv file.
# This is useful for large databases because you only have to "calculateHistogram" for the enitre database once.
# Significantly reduces runtime
def saveHistograms(folder_path, csv_histogram_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            rgb_hist = calculateHistogram(image, color_space='RGB')
            hsv_hist = calculateHistogram(image, color_space='HSV')
            lab_hist = calculateHistogram(image, color_space='LAB')
            data.append([image_path, rgb_hist.tolist(), hsv_hist.tolist(), lab_hist.tolist()])

    df = pd.DataFrame(data, columns=['image', 'rgb_histogram', 'hsv_histogram', 'lab_histogram'])
    df.to_csv(csv_histogram_path, index=False)


#This function loads the color histogram data from the CSV file
def loadHistograms(csv_histogram_path):
    df = pd.read_csv(csv_histogram_path)
    histograms = {}
    for index, row in df.iterrows():
        filename = row['image']
        rgb_hist = np.array(eval(row['rgb_histogram']), dtype=np.float32)
        hsv_hist = np.array(eval(row['hsv_histogram']), dtype=np.float32)
        lab_hist = np.array(eval(row['lab_histogram']), dtype=np.float32)
        histograms[filename] = {'RGB': rgb_hist, 'HSV': hsv_hist, 'LAB': lab_hist}
    return histograms

# This function finds the Top-K similar images based off on color space and equation.
def findImages(query_image_path, folder_path, k=5, bins=(8, 8, 8), method='euclidean', color_space='RGB', histograms=None):
    query_image = cv2.imread(query_image_path)
    query_hist = calculateHistogram(query_image, bins, color_space)

    if histograms == None: 
      distances = {}
      for filename in os.listdir(folder_path):
          if filename.endswith((".png", ".jpg", ".jpeg")) and filename != os.path.basename(query_image_path):
              image_path = os.path.join(folder_path, filename)
              image = cv2.imread(image_path)
              hist = calculateHistogram(image, bins, color_space)
              dist = compareHistograms(query_hist, hist, method)
              distances[image_path] = dist
    else: 
      distances = {}
      for filename, hists in histograms.items():
          if filename != query_image_path:
              hist = hists[color_space]
              dist = compareHistograms(query_hist, hist, method)
              distances[filename] = dist
              
    #A higher correlation or intersection value represents a higher simlarity
    if method in ['correlation', 'intersection']:
        sorted_distances = sorted(distances.items(), key=lambda item: item[1], reverse=True)
    else:
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])

    top_k_images = sorted_distances[:k]
    return top_k_images


# This function displays the Top-K images retrieved 
def displayImages(query_image_path, top_images):
  all_images = [query_image_path]
  all_titles = ['Query Image']

  #Create a list with all images and titles
  for i, (image, dist) in enumerate(top_images):
    all_images.append(image)
    all_titles.append(f'Image #{i + 1}\n Dist: {round(top_images[i][1], 2)}\n File: {os.path.basename(top_images[i][0])}')
  
  #Initialize the figure
  fig, axes = plt.subplots(int(len(all_images) / 5), 6, figsize=(25, int(len(all_images) / 2)))
  #fig.suptitle('Top Similar Images', fontsize=16)

  #Plot each image and its title
  for (i, image), ax in zip(enumerate(all_images), enumerate(axes.flatten())):
    ax[1].imshow(cv2.cvtColor(cv2.imread(all_images[i]), cv2.COLOR_BGR2RGB))
    ax[1].set_title(f'{all_titles[i]}', fontsize = 5)
    ax[1].axis('off')

  #Remove blank plots
  for ax in axes.flat[len(all_images):]:
    ax.remove()
  plt.show()   


# This function plots the color histogram of an image
def plotColorHistogram(image_path, color_space):
  #Convert the image according to the color space
  image = cv2.imread(image_path)
  if color_space == 'RGB':
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      title1, title2, title3 = 'Red', 'Green', 'Blue'
  elif color_space == 'HSV':
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    title1, title2, title3 = 'Hue', 'Saturation', 'Value'
  elif color_space == 'LAB':
    title1, title2, title3 = 'L*', 'a*', 'b*'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
  else: 
      raise ValueError("Unknown method: choose from 'HSV', 'RGB', 'LAB'")

  #Extract the histogram for each color space
  if color_space == 'HSV':
    hist1 = cv2.calcHist([image],[0], None, [180], [0,181]) #hue
  else:
    hist1 = cv2.calcHist([image],[0], None, [256], [0,256]) #red, L*

  hist2 = cv2.calcHist([image],[1], None, [256], [0,256]) #green, saturation, a*
  hist3 = cv2.calcHist([image],[2], None, [256], [0,256]) #blue, value, b*

  #Initialize a list of all colors, titles, and histograms for the plot
  all_colors = ['red', 'green', 'blue']
  all_titles = [title1, title2, title3]
  all_hists = [hist1, hist2, hist3]

  #Initialize plots
  fig, axes = plt.subplots(2, 2)

  #Visualize the color histograms
  for row, col, i in zip([0, 0, 1], [0, 1, 0], range(3)):
    axes[row, col].plot(all_hists[i], all_colors[i])
    axes[row, col].set_title(all_titles[i], fontsize = 10)

  #Visualize the image
  axes[1, 1].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), interpolation='none')
  axes[1, 1].set_title(f'Image: {os.path.basename(image_path)}', fontsize = 10)
  plt.subplots_adjust(hspace=0.4)
  plt.show()