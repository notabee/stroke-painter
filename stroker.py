import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os


class DifferentiableSpline:
    def __init__(self, control_points, canvas_size=(128, 128), device='cpu'):
        self.device = device
        self.canvas_size = canvas_size
        self.control_points = control_points.to(device).detach().requires_grad_(True)

    def interpolate_points(self, num_points=100):
        t_values = torch.linspace(0, 1, num_points, device=self.device, requires_grad=True)

        # Ensure exactly 3 control points for Quadratic Bezier curve
        if len(self.control_points) != 3:
            raise ValueError("Need exactly 3 control points for Quadratic Bezier spline interpolation")

        # Define control points for the Quadratic Bezier curve
        p0, p1, p2 = self.control_points

        # Use Quadratic Bezier formula
        points = []
        for t in t_values:
            # Quadratic Bezier formula
            interpolated_point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            points.append(interpolated_point)

        interpolated_points = torch.stack(points) * torch.tensor(self.canvas_size, device=self.device)
        return interpolated_points

    def generate_spline_canvas(self, colors, brush_sizes, num_points=300, test_mode=False):
        colors = colors.to(self.device)
        brush_sizes = brush_sizes.to(self.device)  # Adjust brush size multiplier for more presence

        canvas = torch.zeros((1, 3, *self.canvas_size), device=self.device, requires_grad=True)
        points = self.interpolate_points(num_points=num_points)

        for i, point in enumerate(points):
            X, Y = torch.meshgrid(torch.arange(self.canvas_size[0], device=self.device),
                                  torch.arange(self.canvas_size[1], device=self.device))
            dist_squared = (X - point[0]) ** 2 + (Y - point[1]) ** 2
            gaussian_mask = torch.exp(-dist_squared / (0.005 * brush_sizes[0] ** 2))  # Sharper strokes

            if test_mode:
                print(f"Dist Squared Min: {dist_squared.min().item()}, Max: {dist_squared.max().item()}")
                print(f"Gaussian Mask Min: {gaussian_mask.min().item()}, Max: {gaussian_mask.max().item()}")

            color = colors[i % len(colors)]
            colored_brush = gaussian_mask * color.view(3, 1, 1)

            # Instead of blending, take the max at each pixel to layer the strokes
            canvas = torch.max(canvas, colored_brush.unsqueeze(0))

        if test_mode:
            print(f"Canvas Min: {canvas.min().item()}, Max: {canvas.max().item()}")
        return torch.clamp(canvas, 0, 1)  # Clamp final result to [0, 1]

    def show_canvas(self, canvas):
        canvas_np = canvas.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        plt.imshow(canvas_np)
        plt.axis('off')
        plt.show()


class StrokePainter(nn.Module):
    def __init__(self, total_strokes, num_control_points_per_stroke=4, hidden_dim=256, n_heads=8, n_enc_layers=3, n_dec_layers=3, canvas_size=(128, 128)):
        super(StrokePainter, self).__init__()
        self.total_strokes = total_strokes
        self.num_control_points_per_stroke = num_control_points_per_stroke
        self.canvas_size = canvas_size

        # Encoder for the input image and canvas
        self.enc_img = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.enc_canvas = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # Convolution layer to combine image and canvas encodings
        self.conv = nn.Conv2d(256, hidden_dim, kernel_size=1)

        # Transformer to generate stroke parameters
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_enc_layers, num_decoder_layers=n_dec_layers)

        # Linear layers to generate stroke parameters from transformer output
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2 * num_control_points_per_stroke + 4)
        )

        # Positional embedding for transformer
        self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))

    def forward(self, img, canvas):
        batch_size, _, H, W = img.shape

        # Encode target image and canvas
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)

        # Combine encoded features and apply convolution
        combined_feat = torch.cat([img_feat, canvas_feat], dim=1)
        combined_feat = self.conv(combined_feat).flatten(2)
        combined_feat = combined_feat.permute(2, 0, 1)

        # Transformer input
        hidden_state = self.transformer(combined_feat, self.query_pos.unsqueeze(1).repeat(1, batch_size, 1))
        hidden_state = hidden_state.permute(1, 0, 2)

        # Generate stroke parameters
        stroke_params = self.linear_param(hidden_state)

        # Split stroke parameters into interpretable parts
        control_points = torch.sigmoid(stroke_params[:, :, :2 * self.num_control_points_per_stroke])
        colors = torch.sigmoid(stroke_params[:, :, 2 * self.num_control_points_per_stroke:2 * self.num_control_points_per_stroke + 3])
        brush_sizes = torch.sigmoid(stroke_params[:, :, -1]) * 20

        # Reshape control points
        control_points = control_points.view(batch_size, self.total_strokes, self.num_control_points_per_stroke, 2)

        # Initialize the canvas and overlap count for each batch item
        accumulated_canvas = torch.zeros((batch_size, 3, *self.canvas_size), device=img.device)
        overlap_count = torch.zeros((batch_size, 1, *self.canvas_size), device=img.device)  # 1-channel for counting overlaps

        for b in range(batch_size):
            for i in range(self.total_strokes):
                # Use control points directly without offsets
                positioned_control_points = control_points[b, i]
                color = colors[b, i]
                brush_size = brush_sizes[b, i]

                # Create the spline for this stroke
                spline = DifferentiableSpline(
                    control_points=positioned_control_points,
                    canvas_size=self.canvas_size,
                    device=img.device
                )

                # Render this stroke on a temporary canvas
                stroke_canvas = spline.generate_spline_canvas(color.unsqueeze(0), brush_sizes=torch.full((100,), brush_size.item(), device=img.device), num_points=100)

                # Accumulate the colors and update the overlap count
                accumulated_canvas[b] += stroke_canvas[0]

        # Clamp the final canvas to ensure pixel values are between 0 and 1
        final_canvas = accumulated_canvas

        return final_canvas
    
    
# Define function to save generated and target images for each epoch
def visualize_and_save_results(images, generated_canvas, save_dir="output_images", epoch=1, batch=1, num_samples=5):
    # Ensure the save directory for the current epoch exists
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Denormalize images and canvases for visualization
    images = (images * 0.5 + 0.5).clamp(0, 1)  # Convert to [0, 1]
    generated_canvas = (generated_canvas * 0.5 + 0.5).clamp(0, 1)  # Convert to [0, 1]

    # Randomly select samples
    samples = random.sample(range(images.size(0)), min(num_samples, images.size(0)))
    
    # Set up plot
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))

    for i, idx in enumerate(samples):
        # Display target image
        axes[0, i].imshow(images[idx].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("Target Image")
        axes[0, i].axis('off')

        # Display generated canvas
        axes[1, i].imshow(generated_canvas[idx].permute(1, 2, 0).cpu().detach().numpy())
        axes[1, i].set_title("Generated Canvas")
        axes[1, i].axis('off')

    # Save the figure for this epoch and batch
    save_path = os.path.join(epoch_dir, f"epoch_{epoch}_batch_{batch}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

    print(f"Saved samples for Epoch {epoch}, Batch {batch} to {save_path}")
    
    
    
torch.cuda.empty_cache()
# Define custom loss weights and options
class TrainingOptions:
    lambda_pixel = 10.0  # Weight for pixel-level loss
    lr = 0.001  # Learning rate
    n_epochs = 100  # Total epochs
    lr_decay_iters = 20  # Decay step for learning rate
    epoch_count = 1  # Start epoch count
    lr_policy = 'step'  # Learning rate policy

opt = TrainingOptions()

image_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StrokePainter(total_strokes=128, num_control_points_per_stroke=3, hidden_dim=256, canvas_size=(image_size, image_size))

# Wrap the model with DataParallel for multi-GPU usage
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)

model.to(device)
criterion_pixel = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.3 ** max(0, (epoch + opt.epoch_count - opt.n_epochs) // 5))

# Load dataset with normalization to [0, 1] and define dataloader
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])
dataset = ImageFolder('1/resized', transform=transform)
dataloader = DataLoader(dataset, batch_size=8*25, shuffle=True, num_workers=4)

# Training loop with progress monitoring
for epoch in range(opt.n_epochs):
    epoch_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.n_epochs}") as pbar:
        for batch_idx, (images, _) in enumerate(pbar, start=1):
            images = images.to(device)

            # Initialize blank canvas with values normalized to the model’s expected range ([-1, 1])
            batch_size = images.size(0)
            blank_canvas = torch.zeros(batch_size, 3, image_size, image_size, device=device)

            # Forward pass
            optimizer.zero_grad()
            generated_canvas = model(images, blank_canvas)

            # Compute loss
            pixel_loss = criterion_pixel(generated_canvas, images) * opt.lambda_pixel
            total_loss = pixel_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate and display batch loss in tqdm
            epoch_loss += total_loss.item()
            pbar.set_postfix({"Batch Loss": total_loss.item()})
            visualize_and_save_results(images, generated_canvas, save_dir="results", epoch=epoch+1, batch=batch_idx, num_samples=5)

    # Adjust learning rate
    scheduler.step()

    # Print epoch loss
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{opt.n_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    # Display random samples of target and generated canvases
    visualize_results(images, generated_canvas, num_samples=5)

    # Save model checkpoints every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"stroke_painter_epoch_{epoch+1}.pth")

print("Training completed.")