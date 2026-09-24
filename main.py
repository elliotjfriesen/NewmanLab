import torch
from PIL import Image
from torchvision.transforms import v2

transform = v2.Compose([
              v2.Resize(560, interpolation=v2.InterpolationMode.BICUBIC),
              v2.CenterCrop(560),
              v2.ToTensor(),
              v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

batch_size = 1
imgs_tensor = torch.zeros(batch_size, 3, 560, 560)

for i in range(batch_size):
    img = Image.open('./test1.jpg')
    print(img.size)
    imgs_tensor[i] = transform(img)

#load large DinoV2 model
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dino.cuda()

#inference
with torch.no_grad():
  features_dict = dino.forward_features(imgs_tensor.cuda())
  features = features_dict['x_norm_patchtokens']
  
  print(features.shape)
  
features = features.squeeze(0)
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

features = features.cpu()
pca = PCA(n_components=1)
scaler = MinMaxScaler()
pca.fit(features)
pca_features = pca.transform(features)
norm_features = scaler.fit_transform(pca_features)


#threshold background
threshold = 0.5 #adjust the threshold based on your images
background = norm_features > threshold

#set background of features to zero
bg_features = features.clone() #make a copy of features
for i in range(bg_features.shape[-1]):
  bg_features[:,i][background[:,0]] = 0

#fit 3 components
pca3 = PCA(n_components=3)
pca3.fit(bg_features)
features_foreground = pca3.transform(bg_features)
norm_features_foreground = scaler.fit_transform(features_foreground)

fig, axes = plt.subplots(3, 4, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    # pca.components_ holds the eigen-vectors/eigenfaces
    ax.imshow(pca.components_[i].reshape(854, 1200), cmap="gray")
    ax.set_title(f"PC {i+1}")
    ax.axis("off")
plt.tight_layout()
plt.show()