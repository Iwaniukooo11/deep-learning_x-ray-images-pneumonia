#write me a function with path (to a folder) as parameter and which will make a for loop on every .jpeg image and then - resize it to 64x64, . On the output I want a dataframe with 100 columns for every component and rows for every image. The function should return the dataframe.. And also 1 additional column: if image is from NORMAL folder then 0, otherwise 1. Becuase path is to a folder, you can assume that there are only 2 folders: NORMAL and PNEUMONIA.. Remember - the most important factor is for loop on every image in the folder. i dont want a pipeline, but a for loop function





def images_to_df(images,type=0):
    print('start')
    import os
    from skimage import io, img_as_ubyte, color
    from skimage.transform import resize
    import numpy as np
    import pandas as pd
    import os

    def process_image(image, size):
        image = resize(image, (max(image.shape), max(image.shape)))
        # image = color.rgb2gray(image)
        image = img_as_ubyte(image)
        image = resize(image, size)
        return image

    
    print(images)
    size = (512, 512)
    df = pd.DataFrame()
    for image in images:
        # print('yoo',image)
        image = process_image(image, size)
        df = df._append(pd.Series(image.flatten()), ignore_index=True)
    df['label'] = type
    return df/255.0

#print me list of files in the folder where this file is
# print(os.listdir('../'))

# df = images_to_df('data/test/NORMAL/*.jpeg')






