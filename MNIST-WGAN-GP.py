import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.datasets import mnist
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



class WGAN_GP():
    def __init__(self, latent_dim, critic, generator, beta):
        super(WGAN_GP, self).__init__()
        self.latent_dim = latent_dim
        self.critic = critic
        print("\ncritic:")
        print(self.critic.summary())

        self.generator = generator
        print("\ngenerator:")
        print(self.generator.summary())

        self.beta = beta

    def compile(self, g_optimizer,d_optimizer):
        print("Compiling the WGAN_GP...")
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.generator.compile(loss=self.wasserstein_loss,optimizer=self.g_optimizer)
        self.critic.compile(loss=self.wasserstein_loss,optimizer=self.d_optimizer)
    

    def interpolate(self,generated_images,real_images):
        alpha = np.random.uniform(0.0,1.0,(generated_images.shape[0],))
        return alpha[:,None,None,None]*real_images + (1.0-alpha[:,None,None,None])*generated_images


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
        
    def gradient_penalty(self,interpolated_images):
        with tf.GradientTape() as tape:
            predictions = self.critic(interpolated_images)
        input_gradients = tape.gradient(predictions, interpolated_images)
        print(input_gradients.shape)
        input_gradients_square = K.square(input_gradients)
        sum_input_gradients_square = K.sum(input_gradients_square,axis=(1,2,3))
        print(sum_input_gradients_square.shape)
        input_gradients_l2_norm = K.sqrt(sum_input_gradients_square)
        return K.mean(input_gradients_l2_norm)


    def train_critic(self,x,y,steps,batch_size):
        real_images, _ = x,y
        for _ in range(steps):
            # Sample random points in the latent space
            batch_size = tf.shape(real_images)[0]
            random_latent_vectors = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Decode the noise to fake images.
            generated_images = self.generator(random_latent_vectors)

            # Combine them with real images. 
            combined_images = tf.concat([generated_images, real_images], axis=0)

            # Assemble labels discriminating real from fake images.
            labels = tf.concat([tf.ones((batch_size, 1)), (-1)*tf.ones((batch_size, 1))], axis=0)

            # Interpolate images for the regularization term ######
            interpolated_images = tf.Variable(self.interpolate(generated_images,real_images),dtype=tf.float32)

            # Train the critic.
            with tf.GradientTape() as tape:
                predictions = self.critic(combined_images)
                d_w_loss = self.wasserstein_loss(labels, predictions)
                d_loss = d_w_loss + self.beta*self.gradient_penalty(interpolated_images)
                
            grads = tape.gradient(d_loss, self.critic.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        
        return d_loss


    def train_generator(self,steps,batch_size):
        for _ in range(steps):
        
            random_latent_vectors = np.random.normal(0, 1, (batch_size, self.latent_dim))
            misleading_labels = (-1)*tf.ones((batch_size, 1))
            
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors)
                predictions = self.critic(fake_images)
                g_loss = self.wasserstein_loss(misleading_labels, predictions)
            
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return g_loss


    def train(self,epochs,x,y, batches_per_epoch):
        loss_dict = {'g_loss':[],'d_loss':[]}
        batch_size=32

        (critic_train_steps,generator_train_steps) = batches_per_epoch

        def batch_generator( x, y, batch_size):
            while True:
                idx = random.sample(range(x.shape[0]),batch_size)
                batch_x = x[idx,::]
                batch_y = y[idx,:]

                yield ( batch_x, batch_y )

        data_gen = batch_generator(x=x,y=y,batch_size=batch_size)
        for e in range(epochs):
            batch_x,batch_y = next(data_gen)
            d_loss = self.train_critic(batch_x,batch_y,steps=critic_train_steps,batch_size=batch_size)
            g_loss = self.train_generator(steps=generator_train_steps,batch_size=batch_size)
            print("[",str(e),"] d_loss:","{0:.7f}".format(d_loss), "g_loss:","{0:.7f}".format(g_loss))
            loss_dict['g_loss'].append(g_loss)
            loss_dict['d_loss'].append(d_loss)

            if e % 25 == 0:
                self.sample_the_generator(e)
                self.plot_metrics(loss_dict,file_name='training_history')
            
            self.generator.save('generator.hdf5')
        
    def sample_the_generator(self,epoch):
        num_samples = 25
        # Sample random points in the latent space.
        random_latent_vectors = np.random.normal(0, 1, (num_samples, self.latent_dim))

        fake_images = K.get_value(self.generator(random_latent_vectors))
        plt.cla()
        plt.figure(figsize=(25,25))
        plt.title("Synthesized images at epoch "+str(epoch))
        for i in range(num_samples):
            plt.subplot(5,5,i+1)
            plt.imshow(fake_images[i,::].reshape(28,28),cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.savefig("fake-samples/"+str(epoch)+".png")
        plt.savefig("current-fake-samples.png")
        plt.close()
    
    def plot_metrics(self,loss_dict, file_name):
        plt.cla()
        plt.figure()
        ax = plt.subplot(1,1,1)
        for i, metric in enumerate(loss_dict.keys()):
            y = loss_dict[metric]
            x = range(len(y))
            ax.plot(x,y, label=metric)
            
        ax.set_xlabel('epoch')
        ax.legend(loss_dict.keys(), loc='upper left')
        plt.tight_layout()
        plt.savefig(file_name+".png")
        plt.close()





### Loading the Dataset ####

(x_train, y_train), (_, _) = mnist.load_data()

num_classes = int(np.max(y_train)+1)
print("There are ",num_classes, " classes in MNIST dataset.")

x_train = (x_train/255.0).astype('float32')
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
print("training dataset: ", x_train.shape)

input_shape = x_train.shape[1:]
y_train = keras.utils.to_categorical(y_train,num_classes)


##### Creating the WGAN_GP ######
latent_dim = 100

def make_generator(latent_dim):
    generator = keras.models.Sequential()
    generator.add(keras.layers.Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    generator.add(keras.layers.Reshape((7, 7, 128)))
    generator.add(keras.layers.UpSampling2D())
    generator.add(keras.layers.Conv2D(128, kernel_size=4, padding="same"))
    generator.add(keras.layers.BatchNormalization(momentum=0.8))
    generator.add(keras.layers.Activation("relu"))
    generator.add(keras.layers.UpSampling2D())
    generator.add(keras.layers.Conv2D(64, kernel_size=4, padding="same"))
    generator.add(keras.layers.BatchNormalization(momentum=0.8))
    generator.add(keras.layers.Activation("relu"))
    generator.add(keras.layers.Conv2D(1, kernel_size=4, padding="same"))
    generator.add(keras.layers.Activation("sigmoid"))
    return generator


def make_critic():
    critic = keras.models.Sequential()
    critic.add(keras.layers.Conv2D(16, kernel_size=3, strides=2, input_shape=(28,28,1), padding="same"))
    critic.add(keras.layers.LeakyReLU(alpha=0.2))
    critic.add(keras.layers.Dropout(0.25))
    critic.add(keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
    critic.add(keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
    critic.add(keras.layers.BatchNormalization(momentum=0.8))
    critic.add(keras.layers.LeakyReLU(alpha=0.2))
    critic.add(keras.layers.Dropout(0.25))
    critic.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    critic.add(keras.layers.BatchNormalization(momentum=0.8))
    critic.add(keras.layers.LeakyReLU(alpha=0.2))
    critic.add(keras.layers.Dropout(0.25))
    critic.add(keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same"))
    critic.add(keras.layers.BatchNormalization(momentum=0.8))
    critic.add(keras.layers.LeakyReLU(alpha=0.2))
    critic.add(keras.layers.Dropout(0.25))
    critic.add(keras.layers.Flatten())
    critic.add(keras.layers.Dense(1))
    return critic
    

generator = make_generator(latent_dim)
critic = make_critic()
GAN = WGAN_GP(latent_dim=latent_dim,critic=critic, generator=generator,beta=10)
GAN.compile(d_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005), g_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005))
GAN.train(x=x_train,y=y_train,epochs=3000, batches_per_epoch=(5,1))