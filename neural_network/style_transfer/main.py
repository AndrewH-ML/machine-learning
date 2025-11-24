import tensorflow as tf 
from tensorflow.keras.applications import vgg19
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds 
from PIL import Image
import numpy as np 
import sys

from model import (
    compute_content_cost,
    compute_style_cost,
    total_cost, 
    get_layer_outputs
)

print(sys.executable)
print(sys.version)

IMG_SIZE = 400
# ******************************************** # 
# LOAD VGG19 PRETRAINED MODEL
vgg = vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(400, 400, 3))
vgg.trainable = False 
print("Layer names: \n")
for layer in vgg.layers:
    print('\t', layer.name)

print(vgg.get_layer('block5_conv4').output)

# ******************************************** # 
# LOAD OXFORD FLOWERS 102 DATASET 
ds = tfds.load("oxford_flowers102", split="train", shuffle_files=True)
it = iter(ds)
content_sample = next(it)
temp = next(it)
style_sample = next(it)

content_raw = content_sample["image"]
style_raw = style_sample["image"]

print("Original content shape:", content_raw.shape, type(content_raw))
print("Original style shape:", style_raw.shape)

content_img = np.array(
    Image.fromarray(content_raw.numpy()).resize((IMG_SIZE, IMG_SIZE))
)
style_img = np.array(
    Image.fromarray(style_raw.numpy()).resize((IMG_SIZE, IMG_SIZE))
)
print("content_img shape: ", np.shape(content_img), type(content_img))
print("style_img shape: ", np.shape(style_img))

#transform to (1, 400, 400, 3)
content_img = tf.constant(np.reshape(content_img, ((1,) + content_img.shape)))
style_img   = tf.constant(np.reshape(style_img, ((1, ) + style_img.shape)))

print("Content image tensor shape:", content_img.shape)
print("Style image tensor shape:", style_img.shape)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(content_img[0].numpy().astype("uint8"))
plt.title("Content image (Oxford Flowers)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(style_img[0].numpy().astype("uint8"))
plt.title("Style image (Oxford Flowers)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ******************************************** # 
# INITIALIZE IMAGE TO BE GENERATED 
generated_image = tf.Variable(tf.image.convert_image_dtype(content_img, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image.assign_add(noise)
generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, 
                        clip_value_max=1.0))
print(generated_image.shape)
plt.imshow(generated_image.numpy()[0])
plt.show()

# ******************************************** # 
# LOAD MODEL

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

content_layer = [('block5_conv4', 1)]

layer_stack = STYLE_LAYERS + content_layer

vgg_model_outputs = get_layer_outputs(vgg, layer_stack)

content_target = vgg_model_outputs(content_img)
style_targets = vgg_model_outputs(style_img)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(style_targets,  a_G, STYLE_LAYERS)
        
        # Compute the content cost
        J_content = compute_content_cost(content_target, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)
        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

    return J


# ******************************************** # 
# UTILS 
def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

epochs = 200

for i in range(epochs):
    J = train_step(generated_image)
    if i % 10 == 0:
        print(f"Step {i}, loss = {J.numpy():.4f}")
        tensor_to_image(generated_image).save(f"out_step_{i}.png")

# ******************************************** #
# FINAL PLOT OF RESULTS

final_img = tensor_to_image(generated_image)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(content_img[0].numpy().astype("uint8"))
plt.title("Content")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(style_img[0].numpy().astype("uint8"))
plt.title("Style")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final_img)
plt.title("Stylized (final)")
plt.axis("off")

plt.tight_layout()
plt.show()

final_img.save("out_final.png")
print("Training complete. Final stylized image saved as out_final.png")