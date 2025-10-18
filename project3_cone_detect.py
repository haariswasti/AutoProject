import torch, torchvision
from PIL import Image
from torchvision.transforms import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ConeDataset(torch.utils.data.Dataset):
    def __init__(self, records, transforms=None):
        self.records = records
        self.transforms = transforms
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["img"]).convert("RGB")
        boxes = torch.as_tensor(r["boxes"], dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        if self.transforms: img = self.transforms(img)
        return img, target

def collate(batch): return tuple(zip(*batch))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes=2
)
model.to(DEVICE)

train_records = []
val_records = []

if len(train_records) == 0:
    print("No training data provided. Add your labeled images to train_records list.")
    print("Format: train_records = [{'img': 'path/to/image.jpg', 'boxes': [[x1,y1,x2,y2]]}, ...]")
    print("Model initialized and ready. Exiting without training.")
else:
    tr_tf = torchvision.transforms.Compose([lambda im: F.to_tensor(im)])
    va_tf = torchvision.transforms.Compose([lambda im: F.to_tensor(im)])
    tr_ds = ConeDataset(train_records, transforms=tr_tf)
    va_ds = ConeDataset(val_records, transforms=va_tf)

    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)

    for epoch in range(1, 11):
        model.train()
        for imgs, targets in tr_ld:
            imgs = [im.to(DEVICE) for im in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch:02d} | loss {loss.item():.4f}")

    torch.save(model.state_dict(), "cone_detector.pt")
    print("Saved detector to cone_detector.pt")
