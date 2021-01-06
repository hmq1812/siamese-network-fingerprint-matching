from PIL import Image

from matching import FingerPrintMatching

model = FingerPrintMatching()


image1 = Image.open('0_noise.jpg')
image2 = Image.open('1_original.jpg')
embedding1 = model.get_embedding(image1)
embedding2 = model.get_embedding(image2)

matching = model.calc_euclidean(embedding1, embedding2)
print(matching)