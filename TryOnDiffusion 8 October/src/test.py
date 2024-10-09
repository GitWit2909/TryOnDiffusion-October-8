import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.model import TryOnDiffusion
from src.dataloader import TryOnDataset

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    image = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((image * 255).astype('uint8'))

def test(model, test_dir, output_dir, device, num_samples=5):
    model.eval()

    test_dataset = TryOnDataset(test_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            image, cloth, image_mask, cloth_mask, pose = test_dataset[i]
            
            image = image.unsqueeze(0).to(device)
            cloth = cloth.unsqueeze(0).to(device)
            
            model_input = torch.cat([image, cloth], dim=1)
            output = model(model_input, torch.tensor([0.5]).to(device))  # Placeholder timestep
            
            # Save results
            original_image = tensor_to_image(image)
            cloth_image = tensor_to_image(cloth)
            result_image = tensor_to_image(output)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(original_image)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            ax2.imshow(cloth_image)
            ax2.set_title("Cloth Image")
            ax2.axis('off')
            
            ax3.imshow(result_image)
            ax3.set_title("Try-On Result")
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"result_{i+1}.png"))
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TryOnDiffusion model")
    
    # Add arguments for the model, dataset, and other configurations
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images and garments")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on")

    args = parser.parse_args()

    # Load the model checkpoint
    device = torch.device(args.device)
    model = TryOnDiffusion().to(device)
    
    if os.path.isfile(args.checkpoint):
        print(f"Loading model checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Run the test
    test(model, args.test_dir, args.output_dir, device, num_samples=args.num_samples)

    print(f"Results saved to {args.output_dir}")
