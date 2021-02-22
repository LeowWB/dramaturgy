import torch.nn as nn

class SentimentNet(nn.Module): # extends nn.Module
	def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, device, drop_prob=0.5):
		super(SentimentNet, self).__init__()
		self.output_size = output_size
		self.n_layers = n_layers
		self.hidden_dim = hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
		self.dropout = nn.Dropout(0.2)
		self.fc = nn.Linear(hidden_dim, output_size)
		self.sigmoid = nn.Sigmoid()
		self.device = device

	def forward(self, x, hidden):
		batch_size = x.size(0)
		x = x.long()	# TODO what's this?
		embeds = self.embedding(x)
		lstm_out, hidden = self.lstm(embeds, hidden)
		lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

		out = self.sigmoid(
			self.fc(
				self.dropout(
					lstm_out
				)
			)
		).view(batch_size, -1)[:,-1]

		return out, hidden

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
						weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
		return hidden
