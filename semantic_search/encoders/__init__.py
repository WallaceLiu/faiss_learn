from .embeddings import BertSentEncoder, EmbeddingEncoder

class EncoderFactory(object):
    
    def __init__(self):
        self.mapping = { "bert_sent": BertSentEncoder()}
    
    
    def get_encoder(self, type_: str) -> EmbeddingEncoder:
        type_ = type_.lower()
        
        if type in self.mapping:
            return self.mapping[type_]
        
        else:
            raise NameError(f"Error: unsupported embedding."
                            f" Supported embeddings: {self.mapping.keys()} ") 
