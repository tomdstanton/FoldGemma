# FoldGemma

FoldGemma is an ultra-fast structural representation model based on the Gemma transformer architecture.





## Examples

### Generate embedding

```python
import torch
from foldgemma.api import FoldGemmaInference
from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.data.vocabulary import Protein3diVocabulary

config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
inference = FoldGemmaInference(config)
vocab = Protein3diVocabulary()

# 1. Your raw amino acid sequence as a byte-string
seq = b"MVLTIY"

# 2. Safely encode the byte-string
tokens = vocab.encode_bytes(seq)

# 3. Convert to a PyTorch tensor and add the batch dimension
tensor = torch.tensor([tokens], dtype=torch.long)

# 4. Derive the bidirectional contextual embeddings
embeddings = inference.encode(tensor)
```