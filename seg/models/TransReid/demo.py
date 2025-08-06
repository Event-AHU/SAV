import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.metrics import euclidean_distance

def load_model(config_file, weights_path):
    """Load the pre-trained model"""
    # Load configuration
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = make_model(cfg, num_class=0, camera_num=0, view_num=0)  # Dummy values for testing
    model.load_param(weights_path)
    model.to(device)
    model.eval()
    
    return model, device, cfg

def load_gallery_features(gallery_dir, model, device, cfg):
    """Load or compute features for gallery images"""
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    # Check if pre-computed features exist
    if os.path.exists('gallery_features.pt'):
        print("Loading pre-computed gallery features...")
        gallery_data = torch.load('gallery_features.pt')
        return gallery_data['features'], gallery_data['paths']
    
    print("Computing gallery features...")
    gallery_features = []
    gallery_paths = []
    
    # Recursively find all images in gallery directory
    for root, dirs, files in os.walk(gallery_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                gallery_paths.append(img_path)
                
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                
                # Extract features
                with torch.no_grad():
                    feat = model(img)
                
                if cfg.TEST.FEAT_NORM == 'yes':
                    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                
                gallery_features.append(feat.cpu())
    
    # Convert to tensors
    gallery_features = torch.cat(gallery_features, dim=0)
    
    # Save computed features
    torch.save({
        'features': gallery_features,
        'paths': gallery_paths
    }, 'gallery_features.pt')
    
    return gallery_features, gallery_paths

def query_image(query_img_path, model, gallery_features, gallery_paths, device, cfg, top_k=5):
    """Query with a target image and retrieve similar images"""
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    # Load and preprocess query image
    query_img = Image.open(query_img_path).convert('RGB')
    query_img_tensor = transform(query_img).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        query_feat = model(query_img_tensor)
    
    if cfg.TEST.FEAT_NORM == 'yes':
        query_feat = torch.nn.functional.normalize(query_feat, dim=1, p=2)
    
    # Compute distances
    dist = euclidean_distance(query_feat.cpu(), gallery_features)
    dist = dist.numpy().flatten()
    
    # Get top-k matches
    indices = np.argsort(dist)[:top_k]
    match_paths = [gallery_paths[idx] for idx in indices]
    match_distances = [dist[idx] for idx in indices]
    
    return match_paths, match_distances

def display_results(query_img_path, match_paths, match_distances, top_k=5):
    """Display the query image and top matches"""
    plt.figure(figsize=(15, 8))
    
    # Display query image
    query_img = Image.open(query_img_path).convert('RGB')
    plt.subplot(1, top_k+1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    
    # Display matches
    for i, (path, dist) in enumerate(zip(match_paths, match_distances)):
        img = Image.open(path).convert('RGB')
        plt.subplot(1, top_k+1, i+2)
        plt.imshow(img)
        plt.title(f"Match {i+1}\nDist: {dist:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('query_results.jpg')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle ReID Query")
    parser.add_argument("--config_file", default="configs/VeRi/vit_transreid_stride.yml", help="path to config file", type=str)
    parser.add_argument("--weights", default="output/transformer_120.pth", help="path to model weights", type=str)
    parser.add_argument("--gallery_dir", required=True, help="path to your gallery image directory", type=str)
    parser.add_argument("--query_img", required=True, help="path to query image", type=str)
    parser.add_argument("--top_k", default=5, help="number of similar images to retrieve", type=int)
    args = parser.parse_args()
    
    # Load model
    model, device, cfg = load_model(args.config_file, args.weights)
    
    # Load or compute gallery features
    gallery_features, gallery_paths = load_gallery_features(args.gallery_dir, model, device, cfg)
    
    # Query with target image
    match_paths, match_distances = query_image(
        args.query_img, model, gallery_features, gallery_paths, device, cfg, args.top_k
    )
    
    # Display results
    display_results(args.query_img, match_paths, match_distances, args.top_k)
    
    # Print results
    print(f"Top {args.top_k} matches for query image {args.query_img}:")
    for i, (path, dist) in enumerate(zip(match_paths, match_distances)):
        print(f"{i+1}. {path} (distance: {dist:.4f})")