# pymazes-vae
Maze generation with Variational Autoencoders

**How to run the script**

1) Create a virtual environment (Optional)
2) Install the requirements:
`pip install -r requirements.txt`

3) Generate input images to train the variational autoencoder on:
`python augment_maze.py`

4) Train the variational autoencoder:
`python maze_vae.py`

5) By default, the script plots samples from the latent space and prints one random sample to output.gif

**Loading weights without retraining**

Comment the line:

`history = vae.fit(x_train, x_train, epochs=5000, batch_size=128, callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss')], validation_data=(x_test, x_test))
`

and the line:

`plot_history(history)`

**Notes**

Maze size should match variational autoencoder layers architecture.
In augment_maze.py:
`size=36`
In vae_maze.py:
`size = 36 * 3`
