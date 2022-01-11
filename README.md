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

5) By default, the script plots samples from the latent space and print one random sample to output.gif

**Loading weights without retraining**
Comment the line:
`vae.fit(x_train, x_train, epochs=300, batch_size=128, callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss')])`

**Notes**
Maze size should match variational autoencoder layers architecture.
In augment_maze.py:
`size=36`
In vae_maze.py:
`size = 36 * 3`
