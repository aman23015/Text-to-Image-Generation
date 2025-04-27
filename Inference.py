from Gan import*

def load_model(model_class, checkpoint_path, device):
    """
    Loads a PyTorch model from a checkpoint.
    """
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Load trained models
encoder_path = "encoder_epoch_100.pth"
generator_path = "generator_epoch_100.pth"

encoder = load_model(SourceEncoder, encoder_path, device)
generator = load_model(TargetGenerator, generator_path, device)

print("✅ Models Loaded Successfully!")

# Function to Generate Images
def generate_images(generator, encoder, text_embeddings, num_images=5):
    """
    Generates images from text embeddings using the trained generator.
    """
    noise = torch.randn(num_images, 100).to(device)
    with torch.no_grad():
        img_repr = torch.randn(num_images, 128).to(device)  # Use random embeddings for unseen classes
        generated_imgs = generator(img_repr, text_embeddings, noise)
    return generated_imgs

# Function to Display Generated Images
def show_images(images):
    """
    Displays a batch of generated images.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for i, img in enumerate(images):
        img = img.cpu().detach().numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        axes[i].imshow(img)
        axes[i].axis("off")
    plt.show()

# Test on Sample Text Embeddings
test_text_embeddings = torch.randn(5, 64).to(device)  # Random text embeddings for now
generated_images = generate_images(generator, encoder, test_text_embeddings)

# Show generated images
show_images(generated_images)


# Function to Evaluate Generator on Test Loader
def evaluate_generator(generator, encoder, test_loader):
    """
    Evaluates the generator using the test dataset.
    """
    generator.eval()
    for real_imgs, text_embeddings in test_loader:
        real_imgs = real_imgs.to(device)
        text_embeddings = text_embeddings.to(device)
        noise = torch.randn(real_imgs.size(0), 100).to(device)

        with torch.no_grad():
            img_repr = encoder(real_imgs)
            generated_imgs = generator(img_repr, text_embeddings, noise)

        # Show real vs generated images
        print("🔹 Real Images:")
        show_images(real_imgs[:5])  # Show first 5 real images
        print("🔹 Generated Images:")
        show_images(generated_imgs[:5])  # Show first 5 generated images

        break  # Show one batch and exit

# Run evaluation
evaluate_generator(generator, encoder, test_loader)
