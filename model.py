from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, num_class, dropout):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        classifier_dropout = (dropout if dropout is not None else 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(768, num_class)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class SequenceClassification(nn.Module):
    def __init__(self, num_class, pretrained_model, dropout: float = None):
        super().__init__()
        self.num_class = num_class
        self.model = pretrained_model
        self.classifier = ClassificationHead(num_class, dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] #last hidden state
        logits = self.classifier(sequence_output)
        return logits