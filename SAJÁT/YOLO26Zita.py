
class YOLO26n(nn.Module):
    """
    YOLO26 Nano — full architecture in pure PyTorch.
 
    Backbone (CSPDarkNet):
        Layer 0: Conv  [3→16,   k=3, s=2]     P1/2   — stem, halves 640→320
        Layer 1: Conv  [16→32,  k=3, s=2]     P2/4   — halves 320→160
        Layer 2: C3k2  [32→64,  n=1, c3k=F]   P2     — first CSP block
        Layer 3: Conv  [64→64,  k=3, s=2]     P3/8   — halves 160→80
        Layer 4: C3k2  [64→128, n=1, c3k=F]   P3     — saved → concat in neck
        Layer 5: Conv  [128→128,k=3, s=2]     P4/16  — halves 80→40
        Layer 6: C3k2  [128→128,n=1, c3k=T]   P4     — saved → concat in neck
        Layer 7: Conv  [128→256,k=3, s=2]     P5/32  — halves 40→20
        Layer 8: C3k2  [256→256,n=1, c3k=T]   P5     — deepest features
 
    Neck top:
        Layer 9 : SPPF   [256→256, k=5, shortcut=T]  — multi-scale pooling
        Layer 10: C2PSA  [256→256, n=1]               — global attention
 
    Neck FPN (top-down):
        Layer 11: Upsample ×2
        Layer 12: Concat  [11, 6]  → 256+128 = 384 ch
        Layer 13: C3k2    [384→128, n=1, c3k=T]       → small object features
        Layer 14: Upsample ×2
        Layer 15: Concat  [14, 4]  → 128+128 = 256 ch  (note: 128 from 13 via upsample + 128 from layer 4 skip... actually 128+128 with YOLO26n channels scaled)
        Layer 16: C3k2    [256→64, n=1, c3k=T]        → medium object features
 
    Neck FPN (bottom-up):
        Layer 17: Conv    [64→64, k=3, s=2]
        Layer 18: Concat  [17, 13] → 64+128 = 192 ch
        Layer 19: C3k2    [192→128, n=1, c3k=T]       → large object features
        Layer 20: Conv    [128→128, k=3, s=2]
        Layer 21: Concat  [20, 10] → 128+256 = 384 ch
        Layer 22: C3k2    [384→256, n=1, c3k=T]
 
    Head (NMS-free):
        Layer 23: Detect26 on [layer16(64ch), layer19(128ch), layer22(256ch)]
 
    Args:
        nc  (int): Number of output classes. Change this for your dataset.
        ch  (int): Input channels. Default 3 (RGB). Change to 1 for grayscale.
 
    Quick customisation points:
        - nc: number of classes (bottom of this file)
        - ch: input channels (first Conv layer)
        - To add a P2 head for tiny objects: add another upsample+concat+C3k2
          after layer 16 and pass 4 feature maps to Detect26.
        - To make it deeper: increase n in any C3k2 block.
        - To swap the attention: replace C2PSA with your own module.
    """
 
    def __init__(self, nc=80, ch=3):
        super().__init__()
        self.nc = nc
 
        # ── BACKBONE ──────────────────────────────────────────────────────────
        # Nano channel widths: width_multiple = 0.25 → base channels scaled down
        # Official YOLO26n uses [16, 32, 64, 128, 256] channels
 
        self.backbone = nn.ModuleDict({
            # Stem: two strided convs to quickly downsample
            'layer0': Conv(ch,  16,  3, 2),         # 640→320, 3→16 ch
            'layer1': Conv(16,  32,  3, 2),         # 320→160, 16→32 ch
 
            # P2 stage: first C3k2, c3k=False (1×1 bottleneck, fast)
            'layer2': C3k2(32,  64,  n=1, c3k=False, e=0.25),
 
            # P3 stage
            'layer3': Conv(64,  64,  3, 2),         # 160→80
            'layer4': C3k2(64,  128, n=1, c3k=False, e=0.25),  # saved for neck
 
            # P4 stage: deeper, c3k=True for larger receptive field
            'layer5': Conv(128, 128, 3, 2),         # 80→40
            'layer6': C3k2(128, 128, n=1, c3k=True),            # saved for neck
 
            # P5 stage: deepest
            'layer7': Conv(128, 256, 3, 2),         # 40→20
            'layer8': C3k2(256, 256, n=1, c3k=True),
        })
 
        # ── NECK TOP ──────────────────────────────────────────────────────────
        # SPPF: multi-scale pooling on deepest features
        self.sppf  = SPPF(256, 256, k=5, n_pools=3, shortcut=True)
        # C2PSA: inject global attention context
        self.c2psa = C2PSA(256, 256, n=1)
 
        # ── NECK FPN: top-down path ───────────────────────────────────────────
        # Upsample + concat with saved backbone features, then C3k2 fusion
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
 
        # After concat(upsample(c2psa_out=256), layer6=128) → 384 ch in
        self.neck_td1 = C3k2(256 + 128, 128, n=1, c3k=True)   # → 128 ch (small obj)
 
        # After concat(upsample(neck_td1=128), layer4=128) → 256 ch in
        self.neck_td2 = C3k2(128 + 128,  64, n=1, c3k=True)   # → 64 ch (medium obj)
 
        # ── NECK FPN: bottom-up path ──────────────────────────────────────────
        # Downsample + concat with top-down features
        self.neck_bu1_conv = Conv(64, 64, 3, 2)                # 64→64, stride 2
        # After concat(bu1_conv=64, neck_td1=128) → 192 ch in
        self.neck_bu1 = C3k2(64 + 128, 128, n=1, c3k=True)    # → 128 ch (large obj)
 
        self.neck_bu2_conv = Conv(128, 128, 3, 2)              # 128→128, stride 2
        # After concat(bu2_conv=128, c2psa_out=256) → 384 ch in
        self.neck_bu2 = C3k2(128 + 256, 256, n=1, c3k=True)   # → 256 ch
 

 # ── HEAD ──────────────────────────────────────────────────────────────
        # Three detection scales:
        #   P3/8  → 64ch  → small objects   (80×80 anchors for 640 input)
        #   P4/16 → 128ch → medium objects  (40×40 anchors)
        #   P5/32 → 256ch → large objects   (20×20 anchors)
        self.head = Detect26(nc=nc, ch=(64, 128, 256))
 
        # Initialise weights
        self._initialize_weights()
 
    def forward(self, x):
        """
        Forward pass.
 
        Args:
            x (Tensor): Input image tensor, shape [B, ch, H, W].
                        Typically [B, 3, 640, 640].
 
        Returns:
            Training:  list of 3 raw tensors (passed to loss function)
            Inference: Tensor [B, nc+4, 8400] — ready for postprocess()
        """
 
        # ── BACKBONE ──────────────────────────────────────────────────────────
        x = self.backbone['layer0'](x)     # [B, 16,  320, 320]
        x = self.backbone['layer1'](x)     # [B, 32,  160, 160]
        x = self.backbone['layer2'](x)     # [B, 64,  160, 160]
        x = self.backbone['layer3'](x)     # [B, 64,   80,  80]
        p3 = self.backbone['layer4'](x)    # [B, 128,  80,  80]  ← saved (skip to neck)
        x  = self.backbone['layer5'](p3)   # [B, 128,  40,  40]
        p4 = self.backbone['layer6'](x)    # [B, 128,  40,  40]  ← saved (skip to neck)
        x  = self.backbone['layer7'](p4)   # [B, 256,  20,  20]
        x  = self.backbone['layer8'](x)    # [B, 256,  20,  20]
 
        # ── NECK TOP ──────────────────────────────────────────────────────────
        x     = self.sppf(x)               # [B, 256,  20,  20]  multi-scale pooling
        neck5 = self.c2psa(x)             # [B, 256,  20,  20]  global attention
 
        # ── NECK FPN: top-down ────────────────────────────────────────────────
        # Upsample deep features and merge with shallower backbone features
        x     = self.upsample(neck5)       # [B, 256,  40,  40]
        x     = torch.cat([x, p4], dim=1) # [B, 384,  40,  40]  concat with layer6
        td1   = self.neck_td1(x)          # [B, 128,  40,  40]  small obj features
 
        x     = self.upsample(td1)        # [B, 128,  80,  80]
        x     = torch.cat([x, p3], dim=1) # [B, 256,  80,  80]  concat with layer4
        td2   = self.neck_td2(x)          # [B, 64,   80,  80]  medium obj features
 
        # ── NECK FPN: bottom-up ───────────────────────────────────────────────
        # Downsample and re-merge to recover large-object context
        x     = self.neck_bu1_conv(td2)   # [B, 64,   40,  40]
        x     = torch.cat([x, td1], dim=1)# [B, 192,  40,  40]  concat with td1
        bu1   = self.neck_bu1(x)          # [B, 128,  40,  40]  large obj features
 
        x     = self.neck_bu2_conv(bu1)   # [B, 128,  20,  20]
        x     = torch.cat([x, neck5], dim=1) # [B, 384, 20, 20]  concat with c2psa
        bu2   = self.neck_bu2(x)          # [B, 256,  20,  20]
 
        # ── HEAD ──────────────────────────────────────────────────────────────
        # Pass 3 feature maps to Detect26:
        #   td2  → P3/8  → 80×80 grid  → small objects
        #   bu1  → P4/16 → 40×40 grid  → medium objects
        #   bu2  → P5/32 → 20×20 grid  → large objects
        return self.head([td2, bu1, bu2])
 
    def _initialize_weights(self):
        """
        Initialise Conv2d weights with Kaiming normal (good for ReLU/SiLU).
        BatchNorm weights set to 1, biases to 0.
        Detection head bias initialised for low initial confidence
        (avoids NaN loss at start of training).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
        # Detection head: initialise class bias so initial sigmoid ≈ 0.01
        # prevents the model from being overconfident at initialisation
        prior_prob = 0.01
        bias_init  = -math.log((1 - prior_prob) / prior_prob)  # ≈ -4.6
        for head in self.head.cv3:
            # last layer in each class head is a Conv2d
            last_conv = list(head.children())[-1]
            if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, bias_init)
 
    def info(self):
        """Print parameter count and layer summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"YOLO26n  |  Total params: {total:,}  |  Trainable: {trainable:,}")
        print(f"Classes: {self.nc}")
        print("-" * 60)
        for name, module in self.named_modules():
            if isinstance(module, (Conv, C3k2, SPPF, C2PSA, Detect26)):
                params = sum(p.numel() for p in module.parameters())
                print(f"  {name:<35} {type(module).__name__:<12} {params:>10,} params")

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
print(model.ckpt["train_args"])