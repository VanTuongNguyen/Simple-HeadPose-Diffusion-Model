from dataset import ZLDataset

dataset = ZLDataset("./dataset/train/")

image, id, text = dataset[0]

print(image.shape)
print(id)
print(text)