# 1. 必要なライブラリのインストール
pip install torch torchvision torchaudio rembg huggingface_hub
pip install "tsr @ git+https://github.com/VAST-AI-Research/TripoSR.git"

# 2. 実行用スクリプトの自動作成
cat << 'EOF' > run_3d.py
import torch
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import rembg

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("モデルをロード中...")
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.to(device)

print("画像処理中 (input.jpg)...")
image = Image.open("input.jpg")
image = remove_background(image, rembg.new_session())
image = resize_foreground(image, 0.85)

print("3Dモデル生成中...")
with torch.no_grad():
    scene_codes = model([image], device=device)
meshes = model.extract_mesh(scene_codes, has_vertex_color=True)
meshes[0].export("output.obj")
print("完了！ 'output.obj' を保存しました。")
EOF

# 3. 実行（※同じフォルダに input.jpg という名前で画像を置いてください）
python run_3d.py
