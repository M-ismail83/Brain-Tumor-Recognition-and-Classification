import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet18, densenet121, mobilenet_v2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from utility import HParams
from model import SimCLR_pl
from sklearn.metrics import classification_report, confusion_matrix

class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_fir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def load_backbone_from_checkpoint(checkpoint_path, device):
    config = HParams(epochs=0)
    base_model_structure = resnet18(weights=None)
    feat_dim = base_model_structure.fc.in_features

    simclr_system = SimCLR_pl.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        model=base_model_structure,
        feat_dim=base_model_structure.fc.in_features,
        map_location=device,
        strict=False
    )

    backbone = simclr_system.model.backbone
    backbone.eval()
    backbone.to(device)

    for param in backbone.parameters():
        param.requires_grad = False

    return backbone

def extract_features(dataloader, model, device):
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = model(images)  # Get features from the backbone
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    features_list = np.concatenate(features_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    return features_list, labels_list


if __name__ == "__main__":
    device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device_name}")

    checkpoint_path = "saved_models/SimCLR_Resnet18_Adam.ckpt"
    labeled_training_set = "evalData/Eval_Training"
    labeled_test_set = "evalData/Eval_Testing"
    batch_size = 21
    img_size = 224

    backbone = load_backbone_from_checkpoint(checkpoint_path, device_name)

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_labeled_dataset = LabeledImageDataset(root_dir=labeled_training_set, transform=eval_transform)
    test_labeled_dataset = LabeledImageDataset(root_dir=labeled_test_set, transform=eval_transform)

    train_feature_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_feature_loader = DataLoader(test_labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    x_train_features, y_train_labels = extract_features(train_feature_loader, backbone, device_name)
    x_test_features, y_test_labels = extract_features(test_feature_loader, backbone, device_name)

    scaler = StandardScaler()
    x_train_features = scaler.fit_transform(x_train_features)
    x_test_features = scaler.transform(x_test_features)

    classifier = LogisticRegression(random_state=0, C=0.1, max_iter=1000, solver='liblinear')
    classifier.fit(x_train_features, y_train_labels)

    y_pred = classifier.predict(x_test_features)
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Logistic Regression Test Accuracy: {accuracy * 100:.2f}%")

    # --- Start of Grad-CAM Visualization ---
    if len(y_test_labels) > 0 and x_test_features.shape[0] > 0:
        print("\n" + "=" * 30)
        print("Grad-CAM Visualization Debugging")
        print("=" * 30)

        num_classes = len(train_labeled_dataset.classes)
        feature_dimension = x_train_features.shape[1]

        # --- Crucial Change: Create a deep copy of the backbone for Grad-CAM ---
        # --- and enable gradients for its parameters. ---
        import copy

        backbone_for_gradcam = copy.deepcopy(backbone)  # backbone is from load_backbone_from_checkpoint
        for param in backbone_for_gradcam.parameters():
            param.requires_grad = True
        backbone_for_gradcam.eval()  # Keep it in eval mode (for BatchNorm, Dropout behavior)


        class GradCamWrapperModel(torch.nn.Module):
            def __init__(self, backbone_model, feat_dim, num_output_classes):
                super().__init__()
                self.backbone = backbone_model  # This will be backbone_for_gradcam
                self.classifier_head = torch.nn.Linear(feat_dim, num_output_classes)

            def forward(self, x):
                # It's good practice to ensure backbone is in eval mode here too
                self.backbone.eval()
                features = self.backbone(x)
                if features.ndim > 2:
                    features = torch.flatten(features, start_dim=1)
                return self.classifier_head(features)


        # Use the modified backbone_for_gradcam
        grad_cam_model = GradCamWrapperModel(backbone_for_gradcam, feature_dimension, num_classes).to(device_name)
        grad_cam_model.eval()  # The wrapper model itself should also be in eval mode

        # Target layer path now refers to layers within backbone_for_gradcam
        # (which is now self.backbone inside GradCamWrapperModel)
        target_layers = [grad_cam_model.backbone.layer4[-1].conv2]
        # As an alternative, you could also try the whole block again if conv2 doesn't work with grads enabled:
        # target_layers = [grad_cam_model.backbone.layer4[-1]]

        print(f"Target layer for Grad-CAM: {target_layers[0]}")
        print(f"Grad-CAM Wrapper Model Structure: \n{grad_cam_model}")

        # ... (the rest of your Grad-CAM try-except block for initialization and image loop) ...
        # Ensure cam_object is initialized within the try-except block as before
        try:
            cam_object = GradCAM(model=grad_cam_model, target_layers=target_layers)
        except Exception as e_init_cam:
            print(f"Error initializing GradCAM object: {e_init_cam}")
            import traceback

            traceback.print_exc()
            cam_object = None

        if cam_object is None:
            print("GradCAM object could not be initialized. Skipping CAM generation.")
        else:
            num_images_to_visualize = min(3, len(test_labeled_dataset))
            if num_images_to_visualize == 0:
                print("No images in test dataset to visualize with Grad-CAM.")

            for i in range(num_images_to_visualize):
                # (Your existing image processing and Grad-CAM generation loop for each image)
                # (Make sure to use the same debugging prints as in the previous good version)
                img_tensor_normalized, true_label_idx = test_labeled_dataset[i]
                input_tensor = img_tensor_normalized.unsqueeze(0).to(device_name)
                predicted_label_idx = y_pred[i]
                targets_for_cam = [ClassifierOutputTarget(predicted_label_idx)]

                print(f"\nProcessing Grad-CAM for image {i}:")
                print(f"  Input tensor shape: {input_tensor.shape}")
                print(f"  Predicted label index for CAM target: {predicted_label_idx}")

                try:
                    with torch.no_grad():  # Test forward pass without affecting gradient context for CAM
                        scores = grad_cam_model(input_tensor)
                        print(
                            f"  Scores from grad_cam_model shape: {scores.shape}, Min: {scores.min().item():.4f}, Max: {scores.max().item():.4f}")
                except Exception as e_fwd:
                    print(f"  Error during grad_cam_model forward pass for image {i}: {e_fwd}")
                    import traceback

                    traceback.print_exc()
                    continue

                grayscale_cam_output = None
                try:
                    # This call needs gradients, so requires_grad should be True on relevant params
                    grayscale_cam_output = cam_object(input_tensor=input_tensor, targets=targets_for_cam)

                    if grayscale_cam_output is None:
                        print(f"  Error: cam_object(input_tensor...) returned None for image {i}.")
                        continue

                    print(
                        f"  Output of cam_object type: {type(grayscale_cam_output)}, shape: {grayscale_cam_output.shape}")
                    grayscale_cam = grayscale_cam_output[0, :]

                    original_img_path, _ = test_labeled_dataset.samples[i]
                    rgb_img_pil = Image.open(original_img_path).convert('RGB')
                    display_transform = transforms.Compose([
                        transforms.Resize(256), transforms.CenterCrop(img_size)
                    ])
                    rgb_img_display_pil = display_transform(rgb_img_pil)
                    rgb_img_display_np = np.array(rgb_img_display_pil) / 255.0

                    visualization = show_cam_on_image(rgb_img_display_np, grayscale_cam, use_rgb=True)
                    true_class_name = test_labeled_dataset.classes[true_label_idx]
                    predicted_class_name = test_labeled_dataset.classes[predicted_label_idx]

                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    axs[0].imshow(rgb_img_display_np);
                    axs[0].set_title(f"Original\nTrue: {true_class_name}");
                    axs[0].axis('off')
                    axs[1].imshow(visualization);
                    axs[1].set_title(f"Grad-CAM\nPred: {predicted_class_name}");
                    axs[1].axis('off')

                    cam_fig_filename = f"grad_cam_img_{i}_true_{true_class_name}_pred_{predicted_class_name}.png"
                    plt.savefig(cam_fig_filename)
                    print(f"  Saved Grad-CAM for image {i} to {cam_fig_filename}")
                    plt.show()

                except Exception as e_cam_gen:
                    print(f"  Error during Grad-CAM generation or processing for image {i}: {e_cam_gen}")
                    if grayscale_cam_output is not None:
                        print(
                            f"  Type of grayscale_cam_output before error: {type(grayscale_cam_output)}, shape: {grayscale_cam_output.shape if hasattr(grayscale_cam_output, 'shape') else 'N/A'}")
                    import traceback

                    traceback.print_exc()
                    # --- End of Grad-CAM Visualization ---

    if len(y_test_labels) > 0:
        print("\n" + "=" * 30)
        print("Detailed Classification Report")
        print("=" * 30)

        # Try to get class names from your dataset object for better readability
        # Assuming your test_dataset object has an attribute like 'classes'
        # which is a list of class names in the order of their labels (0, 1, 2, 3...)
        target_names = None
        if hasattr(test_labeled_dataset, 'classes') and test_labeled_dataset.classes:
            if len(test_labeled_dataset.classes) == 4:  # Check consistency
                target_names = test_labeled_dataset.classes
            else:
                print(f"Warning: Number of classes in dataset ({len(test_labeled_dataset.classes)}) "
                      f"does not match model's num_classes ({backbone.hparams.num_classes}). "
                      "Using generic labels for report.")
                target_names = [f"Class {i}" for i in range(backbone.hparams.num_classes)]

        # Generate and print the classification report
        # zero_division=0 means if a class has no true samples or no predicted samples for a metric,
        # that metric will be 0 instead of raising a warning.
        report = classification_report(
            y_test_labels,
            y_pred,
            target_names=target_names,
            digits=3,  # Number of digits for precision
            zero_division=0
        )
        print(report)

        print("\n" + "=" * 30)
        print("Confusion Matrix")
        print("=" * 30)
        cm = confusion_matrix(y_test_labels, y_pred)
        print(cm)

        if target_names and plt is not None and sns is not None:  # Ensure target_names is defined
            try:
                plt.figure(figsize=(10, 8))  # Adjusted figsize for better layout
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names, yticklabels=target_names,
                            annot_kws={"size": 10})  # Font size for annotations in the heatmap cells
                plt.xlabel('Predicted Label', fontsize=12)
                plt.ylabel('True Label', fontsize=12)

                # Include overall accuracy in the title
                # Ensure 'accuracy' variable is defined from accuracy_score
                if 'accuracy' in locals() or 'accuracy' in globals():
                    title_str = f'Confusion Matrix (Logistic Regression)\nAccuracy: {accuracy * 100:.2f}%'
                else:  # Fallback if accuracy variable name is different or not found
                    title_str = 'Confusion Matrix (Logistic Regression)'
                plt.title(title_str, fontsize=14)

                plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate x-axis labels for better readability
                plt.yticks(rotation=0, fontsize=10)
                plt.tight_layout()  # Adjust layout to prevent labels from overlapping

                cm_filename = "Conclusion_Datas/Resnet18/confusion_matrix_lr.png"
                plt.savefig(cm_filename)
                print(f"Confusion matrix saved to {cm_filename}")
                plt.show()  # Show plot interactively
            except Exception as e_plot:
                print(f"Could not plot confusion matrix: {e_plot}")

        # Optional: Plot confusion matrix
        if target_names and plt is not None and sns is not None:
            try:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names, yticklabels=target_names)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')
                # You might want to save this figure:
                # plt.savefig("confusion_matrix.png")
                plt.show()
            except Exception as e_plot:
                print(f"Could not plot confusion matrix: {e_plot}")

    else:
        print("No predictions were made, cannot generate detailed report. Check test dataset and loader.")


