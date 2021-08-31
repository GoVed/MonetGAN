import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout

from PIL import Image
import glob
 
# define the standalone discriminator model
def define_discriminator(in_shape=(256,256,3)):
    model = Sequential()
    model.add(Conv3D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv3D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the standalone generator model
def define_generator(in_shape=(256,256,3)):
    model = Sequential()
    model.add(Conv3D(256, (4,4), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3D(256, (4,4), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3D(1, (7,7), activation='sigmoid', padding='same'))
    return model
 
# load real images
def load_real_samples():
    X=[]
    images=glob.glob("monet_jpg/*.jpg")
    for image in images:
        X.append(np.array(Image.open(image)))
    X=np.array(X,dtype=np.float32)
    X/=255.0
    return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))


# # create the discriminator
# d_model = define_discriminator()
# # create the generator
# g_model = define_generator()
# # create the gan
# gan_model = define_gan(g_model, d_model)
# # load image data
# dataset = load_real_samples()
# # train model
# train(g_model, d_model, gan_model, dataset, latent_dim)