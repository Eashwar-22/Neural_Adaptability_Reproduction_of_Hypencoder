
from hypencoder_cb.modeling.hypencoder import Hypencoder, HypencoderConfig
import torch

dim = 768
config = HypencoderConfig(
    model_name_or_path="bert-base-uncased",
    base_encoder_output_dim=dim,
    converter_kwargs={
        "vector_dimensions": [dim, 1],
    }
)
model = Hypencoder(config)
print("Weight Shapes:", model.weight_shapes)
print("Bias Shapes:", model.bias_shapes)
print("\nModel Structure:")
print(model)
