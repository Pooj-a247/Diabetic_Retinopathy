import os
import torch
import torch.nn as nn

# --- Dynamic Helper to build the exact DAE layers found in your weights file ---
class DynamicDAEShell(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        # Dynamically build layers based on weights
        for key, value in state_dict.items():
            if key.endswith(".weight"):
                if len(value.shape) == 4:  # Conv2d / ConvTranspose2d
                    out_ch, in_ch, k_h, k_w = value.shape
                    # Check if it's a transpose convolution layer (decoding)
                    if "t_conv" in key or "transpose" in key or "dec" in key:
                        layer = nn.ConvTranspose2d(out_ch, in_ch, kernel_size=k_h, stride=2, padding=1, bias=False)
                    else:
                        layer = nn.Conv2d(in_ch, out_ch, kernel_size=k_h, padding=1, bias=False)
                elif len(value.shape) == 2:  # Linear/Fully Connected
                    out_f, in_f = value.shape
                    layer = nn.Linear(in_f, out_f, bias=False)
                else:
                    continue
                
                self._register_submodule(key[:-7], layer)
                
            elif key.endswith(".bias"):
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
        # Dynamically route through registered children sequentially
        # Assuming standard encoder -> decoder sequential pipeline
        for name, module in self.named_children():
            x = module(x)
        return x

# --- Compilation Orchestration ---
def compile_dae_model():
    weights_path = "dae_retina.pth"
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return

    print("📂 Step 1: Loading raw DAE weights structure...")
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    print("🏗️ Step 2: Dynamically generating matching DAE skeleton shell...")
    model = DynamicDAEShell(state_dict)

    print("📦 Step 3: Loading weights parameters into dynamic DAE skeleton...")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("⚡ Step 4: Compiling DAE to high-speed TorchScript Trace...")
    # Assuming standard grayscale input layout [1 image, 1 channel, 512x512 pixels]
    dummy_input = torch.rand(1, 1, 512, 512)
    
    compiled_model = torch.jit.trace(model, dummy_input, check_trace=False)

    print("💾 Step 5: Saving compiled DAE execution engine...")
    compiled_model.save("dae_retina_compiled.pt")
    print("🎉 Success! Saved as 'dae_retina_compiled.pt'!")

if __name__ == "__main__":
    compile_dae_model()
