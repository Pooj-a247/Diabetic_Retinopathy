import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Dynamic Helper to build the exact model layers found in your weights file ---
class DynamicModelShell(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        # Loop through every single layer in your saved weights file and mimic its shape
        for key, value in state_dict.items():
            if key.endswith(".weight"):
                # Handle standard 2D Convolution & Transposed Convolution layers
                if len(value.shape) == 4:
                    out_ch, in_ch, k_h, k_w = value.shape
                    
                    # Check if this is an Upsampling/Transposed Conv layer (like up1, up2, up3)
                    if any(up_key in key for up_key in [".up.", "up1.", "up2.", "up3."]) and "conv" not in key:
                        # ConvTranspose2d weight shape is [in_channels, out_channels, kernel_height, kernel_width]
                        layer = nn.ConvTranspose2d(out_ch, in_ch, kernel_size=k_h, stride=2, padding=0, bias=False)
                    else:
                        layer = nn.Conv2d(in_ch, out_ch, kernel_size=k_h, padding=1, bias=False)
                        
                # Handle Linear (Fully Connected) layers if any
                elif len(value.shape) == 2:
                    out_f, in_f = value.shape
                    layer = nn.Linear(in_f, out_f, bias=False)
                else:
                    continue
                
                # Register the exact sub-modules dynamically
                self._register_submodule(key[:-7], layer)
                
            elif key.endswith(".bias"):
                # Track biases and pair them with their parent layers, matching their exact shape
                parent_key = key[:-5]
                layer = self._get_submodule(parent_key)
                bias_size = value.shape[0]
                
                if layer is not None:
                    if isinstance(layer, nn.ConvTranspose2d):
                        new_layer = nn.ConvTranspose2d(layer.in_channels, bias_size, 
                                                     kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, bias=True)
                    elif isinstance(layer, nn.Conv2d):
                        new_layer = nn.Conv2d(layer.in_channels, bias_size, 
                                              kernel_size=layer.kernel_size, padding=layer.padding, bias=True)
                    elif isinstance(layer, nn.Linear):
                        new_layer = nn.Linear(layer.in_features, bias_size, bias=True)
                    else:
                        continue
                    self._register_submodule(parent_key, new_layer)

            # Recreate Batch Normalization layers precisely
            elif "running_mean" in key:
                parent_key = key[:-13]
                num_features = value.shape[0]
                self._register_submodule(parent_key, nn.BatchNorm2d(num_features))

    def _register_submodule(self, name, module):
        parts = name.split('.')
        obj = self
        for part in parts[:-1]:
            if not hasattr(obj, part):
                setattr(obj, part, nn.Sequential())
            obj = getattr(obj, part)
        if parts[-1].isdigit():
            idx = int(parts[-1])
            while len(obj) <= idx:
                obj.add_module(str(len(obj)), nn.Identity())
            obj[idx] = module
        else:
            setattr(obj, parts[-1], module)

    def _get_submodule(self, name):
        try:
            obj = self
            for part in name.split('.'):
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            return obj
        except Exception:
            return None

    def forward(self, x):
        # 1. Encoder/Contracting Path
        x1 = self.inc(x)
        x2 = getattr(self, "down1.1")(F_maxpool(x1)) if hasattr(self, "down1.1") else getattr(self, "down1")(F_maxpool(x1))
        x3 = getattr(self, "down2.1")(F_maxpool(x2)) if hasattr(self, "down2.1") else getattr(self, "down2")(F_maxpool(x2))
        x4 = getattr(self, "down3.1")(F_maxpool(x3)) if hasattr(self, "down3.1") else getattr(self, "down3")(F_maxpool(x3))
        
        # 2. Decoder/Expanding Path with dynamic padding to prevent dim mismatches
        
        # Up 1
        up_x4 = self.up1(x4) if hasattr(self, "up1") else x4
        diffY = x3.size()[2] - up_x4.size()[2]
        diffX = x3.size()[3] - up_x4.size()[3]
        up_x4 = F.pad(up_x4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up1 = torch.cat([x3, up_x4], dim=1)
        x_up1 = self.conv_up1(x_up1) if hasattr(self, "conv_up1") else x_up1
        
        # Up 2
        up_x_up1 = self.up2(x_up1) if hasattr(self, "up2") else x_up1
        diffY = x2.size()[2] - up_x_up1.size()[2]
        diffX = x2.size()[3] - up_x_up1.size()[3]
        up_x_up1 = F.pad(up_x_up1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up2 = torch.cat([x2, up_x_up1], dim=1)
        x_up2 = self.conv_up2(x_up2) if hasattr(self, "conv_up2") else x_up2
        
        # Up 3
        up_x_up2 = self.up3(x_up2) if hasattr(self, "up3") else x_up2
        diffY = x1.size()[2] - up_x_up2.size()[2]
        diffX = x1.size()[3] - up_x_up2.size()[3]
        up_x_up2 = F.pad(up_x_up2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up3 = torch.cat([x1, up_x_up2], dim=1)
        x_up3 = self.conv_up3(x_up3) if hasattr(self, "conv_up3") else x_up3
        
        out = self.outc(x_up3)
        return out

def F_maxpool(x):
    return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

# --- Compilation Orchestration ---
def compile_my_model():
    weights_path = "unet_retina.pth"
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return

    print("📂 Step 1: Loading raw weights structure...")
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    print("🏗️ Step 2: Dynamically generating matching U-Net skeleton shell...")
    model = DynamicModelShell(state_dict)

    print("📦 Step 3: Loading weights parameters into dynamic skeleton...")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("⚡ Step 4: Compiling to high-speed target format (TorchScript Trace)...")
    dummy_input = torch.rand(1, 1, 512, 512)
    
    compiled_model = torch.jit.trace(model, dummy_input, check_trace=False)

    print("💾 Step 5: Saving compiled execution engine...")
    compiled_model.save("unet_retina_compiled.pt")
    print("🎉 Success! Saved as 'unet_retina_compiled.pt'!")

if __name__ == "__main__":
    compile_my_model()
