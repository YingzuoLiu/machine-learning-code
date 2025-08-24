"""
PyTorch vs TensorFlow 链路对比：
训练 → 导出 → 推理 (inference)
"""

# ==============================
# 1. PyTorch: Train → ONNX → Inference
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim

class TorchRecModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=-1)
        return self.fc(x)


# ---- 训练
num_users, num_items = 1000, 500
user_ids = torch.randint(0, num_users, (256,))
item_ids = torch.randint(0, num_items, (256,))
labels = torch.randint(0, 2, (256,)).float()

model = TorchRecModel(num_users, num_items)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    preds = model(user_ids, item_ids).squeeze()
    loss = loss_fn(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[PyTorch] Epoch {epoch}, Loss = {loss.item():.4f}")

# ---- 导出 ONNX
dummy_user = torch.randint(0, num_users, (1,))
dummy_item = torch.randint(0, num_items, (1,))
torch.onnx.export(
    model,
    (dummy_user, dummy_item),
    "torch_rec.onnx",
    input_names=["user_ids", "item_ids"],
    output_names=["score"],
    dynamic_axes={"user_ids": {0: "batch"}, "item_ids": {0: "batch"}}
)
print("[PyTorch] 导出 ONNX 成功 -> torch_rec.onnx")

# ---- 加载 ONNX 进行推理
import onnxruntime as ort
ort_session = ort.InferenceSession("torch_rec.onnx")
onnx_inputs = {
    "user_ids": dummy_user.numpy(),
    "item_ids": dummy_item.numpy()
}
onnx_out = ort_session.run(None, onnx_inputs)
print("[PyTorch] ONNX 推理结果:", onnx_out)


# ==============================
# 2. TensorFlow: Train → SavedModel → Inference
# ==============================
import tensorflow as tf
from tensorflow.keras import layers, Model

class TFRecModel(Model):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.user_emb = layers.Embedding(num_users, embed_dim)
        self.item_emb = layers.Embedding(num_items, embed_dim)
        self.d1 = layers.Dense(hidden_dim, activation="relu")
        self.d2 = layers.Dense(1)

    def call(self, inputs):
        user_ids, item_ids = inputs
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = tf.concat([u, i], axis=-1)
        return self.d2(self.d1(x))

# ---- 训练
user_ids = tf.random.uniform((256,), maxval=num_users, dtype=tf.int32)
item_ids = tf.random.uniform((256,), maxval=num_items, dtype=tf.int32)
labels = tf.random.uniform((256,), maxval=2, dtype=tf.int32)

tf_model = TFRecModel(num_users, num_items)
tf_model.compile(optimizer="adam",
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
tf_model.fit((user_ids, item_ids), labels, epochs=3, verbose=1)

# ---- 导出 SavedModel
tf_model.save("tf_rec_model")
print("[TensorFlow] 导出 SavedModel 成功 -> tf_rec_model/")

# ---- 加载 SavedModel 推理
loaded = tf.keras.models.load_model("tf_rec_model")
out = loaded((user_ids[:1], item_ids[:1]))
print("[TensorFlow] SavedModel 推理结果:", out.numpy())
