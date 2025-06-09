import torchvision.models as models
import torch.nn as nn

def build_model(pretrained = True, fine_tune = True, num_classes = 10):
    if pretrained:
        print('[Info]: Loading pre-trained weights')
    else:
        print('[Info]: not loading pre-trained weights')
    
    model = models.efficientnet_b4(pretrained = pretrained)

    if fine_tune:
        print('[Info]: fine-tining all layers ...')
        for params in model.parameters():
            params.requires_grad = True
        
    elif not fine_tune:
        print('[Info]: Freezing hidden layers ...')
        for params in model.parameters():
            params.requires_grad = False

    #change the final classification head
    model.classifier[1] = nn.Linear(in_features=1280, out_features= num_classes)
    return model


if __name__ == "__main__":
    import torch

    # Test with pretrained weights and fine-tuning
    model = build_model(pretrained=True, fine_tune=True, num_classes=5)
    print(model)  # Optional: prints full model structure

    # Check output layer shape
    dummy_input = torch.randn(1, 3, 380, 380)  # EfficientNet-B4 expects 380x380
    output = model(dummy_input)
    print("Output shape:", output.shape)

    # Check number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
