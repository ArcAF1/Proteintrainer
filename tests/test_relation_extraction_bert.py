import sys, types
sys.modules.setdefault('torch', types.ModuleType('torch'))
setattr(sys.modules['torch'], 'softmax', lambda x, dim=None: [[0.9,0.1]])
setattr(sys.modules['torch'], 'no_grad', lambda: types.SimpleNamespace(__enter__=lambda *a:None, __exit__=lambda *a, **k:None))
from types import SimpleNamespace
sys.modules['transformers'] = types.ModuleType('transformers')
setattr(sys.modules['transformers'], 'AutoTokenizer', SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(__call__=lambda *a, **k:{'input_ids':[[1,2]]})))
setattr(sys.modules['transformers'], 'AutoModelForSequenceClassification', SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(eval=lambda:self, logits=[[2,1]])))

import importlib
mod = importlib.import_module('src.relation_extraction_bert')
assert hasattr(mod, 'extract_relations') 