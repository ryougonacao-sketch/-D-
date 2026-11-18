# GitHubからリポジトリを取得
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR

# 必要なライブラリのインストール（PyTorchは環境に合わせて事前に導入してください）
pip install -r requirements.txt
python run.py examples/chair.png --output-dir output/ --model-save-format obj
import torch
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import rembg

# 1. モデルのロード（初回実行時にHugging Faceから自動ダウンロードされます）
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.to(device)

# 2. 画像の読み込みと背景除去・前処理
input_image = Image.open("input_photo.jpg")
rembg_session = rembg.new_session()
image = remove_background(input_image, rembg_session)
image = resize_foreground(image, 0.85)

# 3. 3D表現の推論
with torch.no_grad():
    scene_codes = model([image], device=device)

# 4. メッシュ化してOBJファイルとして保存
meshes = model.extract_mesh(scene_codes, has_vertex_color=True)
meshes[0].export("generated_model.obj")

print("generated_model.obj の生成が完了しました！")
