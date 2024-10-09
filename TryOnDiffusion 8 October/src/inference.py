import torch
from model import TryOnDiffusion
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def inference(model, person_image_path, cloth_image_path, device):
    model.eval()
    
    person_image = load_image(person_image_path).to(device)
    cloth_image = load_image(cloth_image_path).to(device)
    
    with torch.no_grad():
        model_input = torch.cat([person_image, cloth_image], dim=1)
        output = model(model_input, torch.tensor([0.5]).to(device))  # Placeholder timestep
    
    # Convert output tensor to image
    output_image = output.squeeze().cpu().permute(1, 2, 0).numpy()
    output_image = (output_image * 0.5 + 0.5).clip(0, 1)
    
    return Image.fromarray((output_image * 255).astype('uint8'))

# Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TryOnDiffusion().to(device)
    model.load_state_dict(torch.load("tryon_diffusion_model.pth"))
    
    person_image_path = "path/to/person/image.jpg"
    cloth_image_path = "path/to/cloth/image.jpg"
    
    result_image = inference(model, person_image_path, cloth_image_path, device)
    result_image.save("try_on_result.jpg")
