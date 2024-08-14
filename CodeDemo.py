from ColorCBIR import *

# -----------
#Display the Top-K similar images:
# 1) Assign Variables
query_image_path = '/Users/janedoe/Documents/folder_name/image_name.jpg'
folder_path = '/Users/janedoe/Documents/folder_name'
# 2) Call findImages() and save the list to a variable. 
# 3) Call displayImages() and use the variable you saved
top_images = findImages(query_image_path, folder_path, k=10, method = 'euclidean', color_space = 'RGB')
displayImages(query_image_path, top_images)
# -----------


# -----------
#Graph color histogram:
# 1) Assign variable 'query_image_path' 
# 2) Call plotColorHistogram()
plotColorHistogram(query_image_path, 'RGB')
# -----------


# -----------
#Save your database's histogram extractions into a list to save on computation time:
# 1) Assign variables 'folder_path', 'query_image_path' and 'csv_histogram_path'
csv_histogram_path = '/Users/janedoe/Documents/histograms.csv'
# 2) Save your histogram extractions to a csv, only run once to obtain the CSV
saveHistograms(folder_path, csv_histogram_path)
# 3) Assign the obtained CSV to a variable
histograms = loadHistograms(csv_histogram_path)
# 4) Display Top-K similar images and assign "histograms" in findImages() to the obtained CSV
top_images = findImages(query_image_path, folder_path, k=10, method = 'euclidean', color_space = 'RGB', histograms=histograms) 
displayImages(query_image_path, top_images)
# -----------