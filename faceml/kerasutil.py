from PIL import Image
from numpy import asarray
from numpy import expand_dims

def load_image(filename):
    try:
        image = Image.open(filename)
        image = resize_image(image,(160,160))
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        return pixels
    except Exception as e:
        print(str(e))
        return None

def resize_image(src_image, size=(128,128)): 
    
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new square background image
    new_image = Image.new("RGB", size)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
  
    # return the resized image
    return new_image        


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    #print("samples:",samples.shape)
    # make prediction to get embedding
    yhat = model.predict(samples)
    #print("yhat:",yhat.shape, yhat[0])
    return yhat[0]


def tostr(*args):
    retval=""
    for i in range(len(args)):
        retval=retval + " " + str(args[i])
    return retval