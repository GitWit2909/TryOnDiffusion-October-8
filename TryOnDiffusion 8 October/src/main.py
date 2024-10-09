from src.preprocessing import verify_preprocessing
from src.model import TryOnDiffusion
from src.dataloader import get_dataloader
from src.train import train
from src.inference import inference
import torch

def main():
    data_dir = "path/to/your/data/directory"
    
    # Verify preprocessing
    if not verify_preprocessing(data_dir):
        print("Preprocessing verification failed. Please check your data.")
        return
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = TryOnDiffusion().to(device)
    
    # Get dataloader
    dataloader = get_dataloader(data_dir)
    
    # Train model
    trained_model = train(model, dataloader, num_epochs=100, device=device)
    
    # Save trained model
    torch.save(trained_model.state_dict(), "tryon_diffusion_model.pth")
    
    # Inference
    person_image_path = "path/to/person/image.jpg"
    cloth_image_path = "path/to/cloth/image.jpg"
    result_image = inference(trained_model, person_image_path, cloth_image_path, device)
    result_image.save("try_on_result.jpg")

if __name__ == "__main__":
    main()
