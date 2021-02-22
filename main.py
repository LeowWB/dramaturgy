import bz2
from collections import Counter
import re
import nltk
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sentimentnet import SentimentNet
import torch.nn as nn

SMALL = False

def main():
	nltk.download('punkt')

	num_train = 800000
	num_test = 200000

	# bz2 is a tar archive compressed w bz2 algorithm
	train_file = bz2.BZ2File('./amazonreviews/train.ft.txt.bz2')
	test_file = bz2.BZ2File('./amazonreviews/test.ft.txt.bz2')

	if SMALL:
		num_train = 1000
		num_test = 500
		train_file = train_file.readlines(1000000)
		test_file = test_file.readlines(100000)
	else:
		train_file = train_file.readlines()
		test_file = test_file.readlines()

	print("Number of training reviews: " + str(len(train_file)))
	print("Number of test reviews: " + str(len(test_file)))

	# before decoding it's all stored as bytes.
	train_file = [x.decode('utf-8') for x in train_file[:num_train]]
	test_file = [x.decode('utf-8') for x in test_file[:num_test]]

	# extract labels. 2nd arg of split() determines number of splits to be done.
	train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
	train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]
	test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
	test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

	del(train_file)
	del(test_file)

	# clean data - replace all digits with 0
	for i in range(len(train_sentences)):
		train_sentences[i] = re.sub('\d', '0', train_sentences[i])

	for i in range(len(test_sentences)):
		test_sentences[i] = re.sub('\d', '0', test_sentences[i])

	# clean data - replace all urls with '<url>'
	replace_urls(train_sentences)
	replace_urls(test_sentences)

	# tokenization
	words = Counter() # dictionary subclass. counts no of times a word appears

	for i, sentence in enumerate(train_sentences):
		train_sentences[i] = []
		for word in nltk.word_tokenize(sentence):
			words.update([word.lower()])
			train_sentences[i].append(word)

	# by this point, words maps each word to the number of times it appears (in all training sentences)
	# each element of train_sentences is an array of the words that appear in that sentence, in order

	# remove all words that only appear once (probably typo)
	words = {k:v for k,v in words.items() if v>1}

	# sort words by appearances (most common first)
	words = sorted(words, key=words.get, reverse=True)

	# create 2 "words" - one to represent padding and one to represent unknown words (incl the typos)
	words = ['_PAD', '_UNK'] + words

	# assign an index to represent each word. create dictionaries to store the mappings (both dirs)
	word2idx = {o:i for i,o in enumerate(words)}
	idx2word = {i:o for i,o in enumerate(words)}

	# convert the words in the sentences to their corresponding indices
	for i, sentence in enumerate(train_sentences):
		train_sentences[i] = [word2idx[word] if word in word2idx else 1 for word in sentence]

	for i, sentence in enumerate(test_sentences):
		test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

	# pad short sentences and shorten long ones, so that all are the same length
	seq_len = 200
	train_sentences = pad_input(train_sentences, seq_len)
	test_sentences = pad_input(test_sentences, seq_len)

	# convert labels (the ground truths) to numpy array
	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)

	# split test set into validation set and test set.
	split_frac = 0.5
	split_id = int(split_frac * len(test_sentences))
	val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
	val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]


	# this is where we actually start working with pytorch
	train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
	val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
	test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

	batch_size = 100 # originally they had 400, but my gpu runs out of space so no choice

	train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
	val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
	test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

	# cuda -> tensor types that use gpu instead of cpu
	if torch.cuda.is_available():	# by right this should try to use gpu, but mine has too little space
		device = torch.device("cuda")
		print("GPU available")
	else:
		device = torch.device("cpu")
		print("GPU unavailable, use CPU")

	# TODO make sense of this part, esp why sample_x and sample_y have their shapes
	dataiter = iter(train_loader)
	sample_x, sample_y = dataiter.next()
	print(sample_x.shape, sample_y.shape)

	vocab_size = len(word2idx)+1
	output_size = 1
	embedding_dim = 400
	hidden_dim = 512
	n_layers = 2
	model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, device)
	model.to(device)
	print(model)

	lr = 0.005
	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	epochs = 2
	counter = 0
	print_every = 1000
	clip = 5
	valid_loss_min = np.Inf	# lowest loss we've seen so far on validation set

	model.train()

	for i in range(epochs):
		h = model.init_hidden(batch_size)

		for inputs, labels in train_loader:
			h = tuple([e.data for e in h])
			inputs, labels = inputs.to(device), labels.to(device)
			model.zero_grad()
			output, h = model(inputs, h)
			loss = criterion(output.squeeze(), labels.float())
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()

			if counter%print_every == 0:
				val_h = model.init_hidden(batch_size)
				val_losses = []
				model.eval()
				for inp, lab in val_loader:
					val_h = tuple([each.data for each in val_h])
					inp, lab = inp.to(device), lab.to(device)
					out, val_h = model(inp, val_h)
					val_loss = criterion(out.squeeze(), lab.float())
					val_losses.append(val_loss.item())

				model.train()
				print("Epoch: {}/{}...".format(i+1, epochs),
					"Step: {}...".format(counter),
					"Loss: {:.6f}...".format(loss.item()),
					"Val Loss: {:.6f}".format(np.mean(val_losses)))
				
				# if we hit a new minimum for loss on validation set then save the current model state
				if np.mean(val_losses) <= valid_loss_min:
					torch.save(model.state_dict(), './state_dict.pt')
					print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
					valid_loss_min = np.mean(val_losses)
			
			counter += 1

		# done training. time to test.
		model.load_state_dict(torch.load('./state_dict.pt'))
		test_losses = []
		num_correct = 0
		h = model.init_hidden(batch_size)

		model.eval()
		for inputs, labels in test_loader:
			h = tuple([each.data for each in h])
			inputs, labels = inputs.to(device), labels.to(device)
			output, h = model(inputs, h)
			test_loss = criterion(output.squeeze(), labels.float())
			test_losses.append(test_loss.item())
			pred = torch.round(output.squeeze()) # round to 0 or 1
			correct_tensor = pred.eq(labels.float().view_as(pred))
			correct = np.squeeze(correct_tensor.cpu().numpy())
			num_correct += np.sum(correct)

			print("Test loss: {:.3f}".format(np.mean(test_losses)))
			print("Test accuracy: {:.3f}%".format(num_correct/len(test_loader.dataset)*100))

# pads short sentences and shortens long ones
def pad_input(sentences, seq_len):
	features = np.zeros((len(sentences), seq_len), dtype=int)
	for ii, review in enumerate(sentences):
		if len(review) != 0:
			features[ii, -len(review):] = np.array(review)[:seq_len]
	return features

# replace all urls with '<url>'
def replace_urls(arr):
	for i in range(len(arr)):
		if 'www.' in arr[i] or 'http:' in arr[i] or 'https:' in arr[i] or '.com' in arr[i]:
			arr[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", arr[i])


main()