def create_image_pipeline(images,type):
    if not images:
        raise ValueError("images list is empty")
    if type not in [0, 1]:
        raise ValueError("type must be either 0 or 1")
    
    
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import Normalizer
    from skimage import io, img_as_ubyte, color
    from skimage.transform import resize
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import make_pipeline
    from skimage.io import imread_collection
    from joblib import dump
    from skimage.exposure import equalize_hist
    from skimage.morphology import binary_erosion
    from skimage.color import rgb2gray



    class ImageProcessor(BaseEstimator, TransformerMixin):
        def __init__(self, size=(512, 512)):
            self.size = size

        def fit(self, X, y=None):
            return self
        
       

        def crop_center(self, img):
            # Convert the image to grayscale if it has more than two dimensions
            print(img.shape)
            if len(img.shape) > 2:
                img = rgb2gray(img)
            y, x = img.shape
            crop_size = min(y, x)
            start_x = x//2 - crop_size//2
            start_y = y//2 - crop_size//2
            return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

        def transform(self, X, y=None):
            X = [self.crop_center(image) for image in X]
            
            X = [resize(image, (max(image.shape), max(image.shape))) for image in X]
            # X = [color.rgb2gray(image) for image in X]
            X = [img_as_ubyte(image) for image in X]
            X = [resize(image, self.size) for image in X]
            X = [image / 255.0 for image in X]  # Normalize pixel values to [0, 1]
             # Add a new column with the same value as type (0 or 1)
            # if type is not None:
            #     X_with_type = [np.c_[image, np.full((image.shape[0], 1), type)] for image in X]
            #     return X_with_type
            # else:
            #     return X
            X = [equalize_hist(image) for image in X]  # Issue fixed
            # X = [binary_erosion(image) for image in X]  # Issue to fix
            return X

    class ImageFlattener(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return pd.DataFrame([image.flatten() for image in X])

    
    

    image_pipeline = make_pipeline(
        ImageProcessor(),
        ImageFlattener()
        
    )
    foo = image_pipeline.fit_transform(images)
    #from foo, create a data frame
    foo = pd.DataFrame(foo)
    #add to it column called "label" with value of type
    foo['label'] = type
    return  foo
    
    


# Save the pipelines
#dump(image_pipeline, 'image_pipeline.joblib')