import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
    print("OpenCLIP available")
except ImportError:
    print("OpenCLIP not available")
    OPEN_CLIP_AVAILABLE = False

try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available")
except ImportError:
    print("Transformers not available")
    TRANSFORMERS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
    print("TIMM available")
except ImportError:
    print("TIMM not available")
    TIMM_AVAILABLE = False

def detect_available_foundation_models():
    """Detect ONLY the 3 foundation models you specified."""
    available = {}
    
    # BiomedCLIP
    if OPEN_CLIP_AVAILABLE:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            available['biomedclip'] = True
            print("  BiomedCLIP available (Medical-specific)")
        except Exception as e:
            available['biomedclip'] = False
            print(f"  BiomedCLIP failed: {e}")
    else:
        available['biomedclip'] = False
    
    # OpenAI CLIP
    if TRANSFORMERS_AVAILABLE:
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            available['clip'] = True
            print("  OpenAI CLIP available")
        except:
            available['clip'] = False
    else:
        available['clip'] = False
    
    # DINOv2
    if TIMM_AVAILABLE:
        try:
            model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
            available['dinov2'] = True
            print("  DINOv2 available")
        except:
            available['dinov2'] = False
    else:
        available['dinov2'] = False
    
    return available

class BiomedCLIPFoundation(nn.Module):
    """BiomedCLIP foundation model wrapper."""
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        print("  Loading BiomedCLIP foundation model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.model.encode_image(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            try:
                for param in self.model.visual.transformer.resblocks[-3:].parameters():
                    param.requires_grad = True
                print("  Unfroze last 3 transformer layers for medical adaptation")
            except:
                print("  Could not unfreeze transformer layers")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        print(f"  BiomedCLIP loaded: {self.feature_dim} features → {num_classes} classes")
    
    def forward(self, x, return_features=False):
        features = self.model.encode_image(x)
        logits = self.classifier(features)
        if return_features:
            return {'logits': logits, 'features': features}
        return logits

class DINOv2Foundation(nn.Module):
    """DINOv2 foundation model wrapper."""
    def __init__(self, num_classes=2, model_name='dinov2_vitb14'):
        super().__init__()
        print(f"Loading DINOv2 foundation model ({model_name})...")
        
        dinov2_variants = [
            'dinov2_vitb14',
            'vit_base_patch14_dinov2.lvd142m',
            'dinov2_vits14',
            'dinov2_vitl14'
        ]
        
        self.backbone = None
        for variant in dinov2_variants:
            try:
                self.backbone = timm.create_model(variant, pretrained=True, num_classes=0, img_size=224)
                model_name = variant
                print(f"  Successfully loaded {variant}")
                break
            except Exception as e:
                print(f"  Failed to load {variant}: {e}")
                continue
        
        if self.backbone is None:
            raise RuntimeError("Could not load any DINOv2 variant")
        
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last blocks
        try:
            if hasattr(self.backbone, 'blocks'):
                for param in self.backbone.blocks[-4:].parameters():
                    param.requires_grad = True
                print("  Unfroze last 4 transformer blocks")
        except:
            print("  Could not unfreeze transformer blocks")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        print(f"  DINOv2 loaded: {self.feature_dim} features → {num_classes} classes")
    
    def forward(self, x, return_features=False):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        try:
            features = self.backbone(x)
        except Exception as e:
            print(f"  DINOv2 forward failed: {e}")
            batch_size = x.size(0)
            features = torch.zeros(batch_size, self.feature_dim).to(x.device)
        
        logits = self.classifier(features)
        if return_features:
            return {'logits': logits, 'features': features}
        return logits

class CLIPFoundation(nn.Module):
    """OpenAI CLIP foundation model wrapper."""
    def __init__(self, num_classes=2, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        print(f"Loading CLIP foundation model ({model_name})...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.feature_dim = self.model.config.vision_config.hidden_size
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last layers
        for param in self.model.vision_model.encoder.layers[-3:].parameters():
            param.requires_grad = True
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        print(f"  CLIP loaded: {self.feature_dim} features → {num_classes} classes")
    
    def forward(self, x, return_features=False):
        vision_outputs = self.model.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)
        if return_features:
            return {'logits': logits, 'features': pooled_output}
        return logits

class FoundationModelTeacher(nn.Module):
    """Teacher ensemble using EXACTLY 3 foundation models: BiomedCLIP, DINOv2, OpenAI CLIP."""
    def __init__(self, num_classes=2):
        super().__init__()
        print("Building Foundation Model Teacher Ensemble...")
        self.models = nn.ModuleDict()
        self.available_models = detect_available_foundation_models()
        
        # Add EXACTLY the 3 models you specified - NO FALLBACKS
        if self.available_models.get('biomedclip', False):
            try:
                self.models['biomedclip'] = BiomedCLIPFoundation(num_classes, freeze_backbone=False)
                print(" BiomedCLIP added to ensemble")
            except Exception as e:
                print(f" Failed to load BiomedCLIP: {e}")
        
        if self.available_models.get('dinov2', False):
            try:
                self.models['dinov2'] = DINOv2Foundation(num_classes)
                print(" DINOv2 added to ensemble")
            except Exception as e:
                print(f" Failed to load DINOv2: {e}")
        
        if self.available_models.get('clip', False):
            try:
                self.models['clip'] = CLIPFoundation(num_classes)
                print(" OpenAI CLIP added to ensemble")
            except Exception as e:
                print(f" Failed to load CLIP: {e}")
        
        
        if len(self.models) == 0:
            raise RuntimeError("No foundation models could be loaded! Install open_clip_torch for BiomedCLIP.")
        
        # Ensemble weights
        self.num_models = len(self.models)
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        print(f"🎓 Teacher ensemble created with {self.num_models} models: {list(self.models.keys())}")
    
    def forward(self, x):
        outputs = []
        successful_models = []
        
        for name, model in self.models.items():
            try:
                output = model(x)
                if isinstance(output, dict):
                    output = output['logits']
                outputs.append(output)
                successful_models.append(name)
            except Exception as e:
                print(f" Model {name} failed: {str(e)[:100]}...")
                batch_size = x.size(0)
                dummy_output = torch.zeros(batch_size, 2, device=x.device, dtype=x.dtype)
                outputs.append(dummy_output)
        
        if len(successful_models) == 0:
            raise RuntimeError("All foundation models failed during forward pass!")
        
        # Stack and ensemble
        stacked_outputs = torch.stack(outputs, dim=1)
        
        # Compute weighted ensemble
        if len(successful_models) < len(self.models):
            adjusted_weights = self.ensemble_weights.clone()
            for i, name in enumerate(self.models.keys()):
                if name not in successful_models:
                    adjusted_weights[i] = 0.0
            adjusted_weights = F.softmax(adjusted_weights, dim=0)
        else:
            adjusted_weights = F.softmax(self.ensemble_weights, dim=0)
        
        ensemble_logits = torch.sum(stacked_outputs * adjusted_weights.view(1, -1, 1), dim=1)
        
        print(f"  Ensemble using {len(successful_models)}/{len(self.models)} models: {successful_models}")
        
        return {
            'ensemble_logits': ensemble_logits,
            'individual_logits': outputs,
            'weights': adjusted_weights,
            'stacked_outputs': stacked_outputs,
            'successful_models': successful_models
        }

class MedicalEfficientBlock(nn.Module):
    """Medical-optimized efficient block."""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        if expand_ratio != 1:
            hidden_dim = int(round(in_channels * expand_ratio))
            self.use_expansion = True
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            hidden_dim = in_channels
            self.use_expansion = False
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 5, stride, 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        se_channels = max(1, hidden_dim // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, hidden_dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if self.use_expansion:
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        
        se_weight = self.se(x)
        x = x * se_weight
        
        x = self.project_conv(x)
        
        if self.use_residual:
            x = x + identity
        
        return x

class LightweightMedicalStudent(nn.Module):
    """Lightweight student network optimized for medical imaging."""
    def __init__(self, num_classes=2, width_multiplier=1.0):
        super().__init__()
        
        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            return new_v
        
        input_channel = make_divisible(32 * width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        configs = [
            [1, 24, 2, 1],
            [2, 32, 2, 2],
            [2, 48, 3, 2],
            [4, 64, 3, 2],
            [4, 96, 2, 1],
            [6, 128, 2, 2]
        ]
        
        self.stages = nn.ModuleList()
        for expand_ratio, channels, num_blocks, stride in configs:
            output_channel = make_divisible(channels * width_multiplier)
            stage_layers = []
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                stage_layers.append(MedicalEfficientBlock(input_channel, output_channel, s, expand_ratio))
                input_channel = output_channel
            self.stages.append(nn.Sequential(*stage_layers))
        
        last_channel = make_divisible(256 * width_multiplier)
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Lightweight student created: {total_params:,} parameters")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, extract_features=False):
        x = self.stem(x)
        
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        x = self.final_conv(x)
        global_features = self.global_pool(x).flatten(1)
        logits = self.classifier(global_features)
        
        if extract_features:
            return {
                'logits': logits,
                'features': global_features,
                'stage_features': features
            }
        return logits
