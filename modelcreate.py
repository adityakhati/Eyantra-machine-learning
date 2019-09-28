
# Homecoming (eYRC-2018): Task 1B
# Fruit Classification with a CNN

#from model import FNet
# import required modules
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



def train_model(dataset_path, debug=False, destination_path='', save=False):

	# Hyper parameters
	device = 'cpu'
	num_epochs = 5
	num_classes = 10
	batch_size = 100
	learning_rate = 0.001

	"""Trains model with set hyper-parameters and provide an option to save the model.

	This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should accept dataset_path which will be path to the dataset directory. You should also specify an option to save the trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

	Args:
		dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
		debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
		destination_path (str, optional): Destination to save the model file. Defaults to ''.
		save (bool, optional): Saves model if True. Defaults to False.

	Returns:
		loss (torch.tensor): Train loss and validation loss.
		accuracy (torch.tensor): Train accuracy and validation accuracy.
	"""
	# Write your code here
	#Dataset
	train_dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms.ToTensor())
	test_dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms.ToTensor())

	#DataLoader
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
	# The code must follow a similar structure

	# Convolutional neural network (two convolutional layers)
	class ConvNet(nn.Module):
		def __init__(self, num_classes=10):
			super(ConvNet, self).__init__()
			self.layer1 = nn.Sequential(
				nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
				nn.BatchNorm2d(16),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2))
			self.layer2 = nn.Sequential(
				nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2))
			self.fc = nn.Linear(7*7*32, num_classes)
			
		def forward(self, x):
			out = self.layer1(x)
			out = self.layer2(out)
			out = out.reshape(out.size(0), -1)
			out = self.fc(out)
			return out
	# NOTE: Make sure you use torch.device() to use GPU if available
	model = ConvNet(num_classes).to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	# Train the model
	total_step = len(train_loader)
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.to(device)
			
			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if (i+1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

	# Test the model
	model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
	if save == True:
		# Save the model checkpoint
		torch.save(model.state_dict(), destination_path)

if __name__ == "__main__":
	#train_model('C:/Users/shwet/Desktop/Task 1B/Data/fruits/', save=True, destination_path='./')
	train_model('C:/Users/DELL/Anaconda3/envs/HC#3190_stage1/ImageToMNIST', save=True, destination_path='./')
	