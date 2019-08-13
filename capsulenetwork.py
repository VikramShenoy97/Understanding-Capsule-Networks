import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from make_graph import plot_graph, generate_reconstructions
import numpy as np
import time

# execfile("make_graph.py") -> For Google Colab

def squash(tensor, dim=-1):
    # Square of Absolute Value
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    # scale = ||Vc||^2 / (1 + ||Vc||^2)
    scale = squared_norm / (1 + squared_norm)
    # squash = scale * (Vc / ||Vc||)
    return scale * tensor / torch.sqrt(squared_norm)

def softmax(tensor, dim=1):
    # Shape of tensor = [10 x 128 x 1152 x 1 x 16]
    transposed_tensor = tensor.transpose(dim, len(tensor.shape)-1)
    # Shape of tensor = [10 x 128 x 16 x 1 x 1152]
    softmaxed_tensor = F.softmax(transposed_tensor.contiguous().view(-1, transposed_tensor.shape[-1]), dim=-1)
    # Shape of softmaxed_tensor = [20480 x 1152]
    output_tensor = softmaxed_tensor.view(*transposed_tensor.shape).transpose(dim, len(tensor.shape)-1)
    # Shape of output_tensor = [10 x 128 x 1152 x 1 x 16]
    return output_tensor

class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsuleLayer, self).__init__()
        # Repeat Convolutions based on number of capsules (8 in this case).
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=256, out_channels=32, kernel_size=9, stride=2, padding=0) for _ in range(num_capsules)])

    def forward(self, x):
        # Shape of x = [128 x 256 x 20 x 20]
        # Apply Convolutions to Input x to generate capsules.
        outputs = [capsule(x).view(x.shape[0], -1, 1) for capsule in self.capsules]
        # [128 x 256 x 20 x 20] --> [128 x (32*6*6) x 1]
        # outputs is now a list of length 8 (number of capsules) with each element as [128 x 1152 x 1]
        outputs = torch.cat(outputs, dim=-1)
        # Shape of outputs = [128 x 1152 x 8]
        outputs = squash(outputs)
        return outputs

class DigitCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, routing_nodes, in_channels, out_channels, routing_iterations=3):
        super(DigitCapsuleLayer, self).__init__()
        self.routing_iterations = routing_iterations
        # Weights = [10 x 1152 x 8 x 16]
        self.weights = torch.randn(num_capsules, routing_nodes, in_channels, out_channels).cuda()

    def forward(self, x):
        # Shape of x = [128 x 1152 x 8]
        # Shape of x = [1 x 128 x 1152 x 1 x 8], Shape of Weights = [10 x 1 x 1152 x 8 x16]
        x_hat = torch.matmul(x[None,:, :, None, :], self.weights[:, None, :, :, :])
        # Shape of x_hat = [10 x 128 x 1152 x 1 x 16]
        # b is a temporary variable that will store the value of routing weights c and will be gradually updated.
        b = torch.zeros(*x_hat.shape).cuda()
        # Dynamic Routing Algorithm
        for i in range(self.routing_iterations):
            # Routing weights for all capsules of layer l (i.e dim_2 = 1152)
            c = softmax(b, dim=2)
            # Weighted sum of x_hat and routing weights c across all capsules of layer l (i.e. dim_2 = 1152)
            outputs = squash((x_hat * c).sum(dim=2, keepdim=True))
            # Shape of outputs = [10 x 128 x 1 x 1 x 16]
            if i != (self.routing_iterations-1):
                # Weight Update Step: Update weight b using dot product similarity.
                db = (x_hat * outputs).sum(dim=-1, keepdim=True)
                # Shape of db = [10 x 128 x 1152 x 1 x 1]
                b = b + db
        return outputs

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0)
        self.primary_capsules = PrimaryCapsuleLayer(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsuleLayer(num_capsules=10, routing_nodes=32*6*6, in_channels=8, out_channels=16, routing_iterations=3)
        self.decoder = nn.Sequential(nn.Linear(16*10, 512), nn.ReLU(inplace=True), nn.Linear(512, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 784), nn.Sigmoid())

    def forward(self, x, y=None, pose=False):
        # Encoder Network
        # Layer 1: Convolutional Layer
        # Input = [128 x 1 x 28 x 28]
        # Output = [128 x 256 x 20 x 20]
        x = F.relu(self.conv1(x), inplace=True)
        # Layer 2: Primary Capsule Layer
        # Input = [128 x 256 x 20 x 20]
        # Output = [128 x (32*6*6) x 8] = [128 x 1152 x 8]
        x = self.primary_capsules(x)
        # Layer 3: Digit Capsule Layer
        # Input = [128 x 1152 x 8]
        # Output = [10 x 128 x 1 x 1 x 16]
        x = self.digit_capsules(x)
        x = x.squeeze().transpose(0, 1)
        # Shape of x = [128, 10, 16]
        class_scores = torch.sqrt((x ** 2).sum(dim=-1))
        # Shape of class_scores = [128 x 10]
        class_probabilities = F.softmax(class_scores, dim=-1)
        # Shape of class_probabilities = [128 x 10]
        if y is None: # During Testing
            # Generate one hot encoded labels from class probabilities
            _, labels = class_probabilities.topk(k=1, dim=1)
            y = torch.eye(10).cuda().index_select(dim=0, index=labels.squeeze())
        if pose == True:
          # Dive deeper into what each of the 16 dimensions represent by fluctuating each value while holding
          #other values constant
          dimension_list = []
          for dim in range(0, x.shape[2]):
            alpha = -0.25
            interval_list = []
            while(alpha <= 0.25):
              temp = x.contiguous().view(1, -1, x.shape[2]).clone()
              # Update one dimension while keeping other dimensions constant.
              temp[0, :, dim] = temp[0, :, dim] + alpha
              alpha = alpha + 0.05
              interval_list.append(temp)
            temp = torch.cat(interval_list, dim=0).unsqueeze(dim=0)
            dimension_list.append(temp)
          x = torch.cat(dimension_list, dim=0)
          x = x.view(x.shape[0], x.shape[1], x.shape[2]/10, 10, x.shape[3])
          repeat_y = x.shape[0] * x.shape[1]
          x = x.view(-1, x.shape[3], x.shape[4])
          y = y.unsqueeze(dim=0)
          y_list = []
          for _ in range(repeat_y):
            y_list.append(y)
          y = torch.cat(y_list, dim=0)
          y = y.view(-1, y.shape[2])
        # Decoder Network
        # Layer 4: Fully Connected Layer 1
        # Input = 16*10 = 160 (x[128 x 10 x 16] * y[128 x 10 x 1](one-hot encoded labels) is done to ignore incorrect vectors through 0 masking)
        # Output = 512
        # Layer 5: Fully Connected Layer 2
        # Input = 512
        # Output = 1024
        # Layer 6: Fully Connected Layer 3
        # Input = 1024
        # Output = 28*28 = 784
        reconstructions = self.decoder((x * y[:, :, None]).view(x.shape[0], -1))
        # Shape of reconstructions = [128 x 784]
        return class_probabilities, reconstructions

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.m_positive = 0.9
        self.m_negative = 0.1
        self.lambda_constant = 0.5
        self.alpha_constant = 0.0005

    def forward(self, images, reconstructions, labels, class_probabilities):
        # Compute Margin Loss
        # In this case ReLU works the same as the Max function used in the original paper i.e. Make sure everything is positive.
        lhs = labels * F.relu(self.m_positive - class_probabilities, inplace=True)**2
        rhs = (1. - labels) * F.relu(class_probabilities - self.m_negative, inplace=True)**2
        margin_loss = lhs + self.lambda_constant*rhs
        # Get scalar value
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.shape[0], -1)
        # Compute Reconstruction Loss
        reconstruction_loss = self.reconstruction_loss(images, reconstructions)
        # Compute Overall Loss
        overall_loss = (margin_loss + self.alpha_constant * reconstruction_loss) / images.shape[0]
        return overall_loss

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST("MNIST_data/", transform=transform, download=True, train=True)
testset = datasets.MNIST("MNIST_data/", transform=transform, download=True, train=False)

model = CapsuleNetwork()
model.cuda()
optimizer = optim.Adam(model.parameters())
loss_function = CapsuleLoss()
epochs = 100
overall_training_accuracy = []
overall_training_loss = []
mode = "Test"
if mode == "Train":
  # Training
  trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=128)
  for epoch in range(epochs):
      start_time = time.time()
      running_loss = 0
      training_accuracy = 0
      count = 0
      percent = 0
      for images, raw_labels in iter(trainloader):
          count += 1
          # Generate One Hot encodings of labels
          labels = torch.eye(10).index_select(dim=0, index=raw_labels)
          class_probabilities, reconstructions = model(images.cuda(), labels.cuda())
          # Compute Loss and Gradients
          loss = loss_function(images.cuda(), reconstructions, labels.cuda(), class_probabilities)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          _, logits = class_probabilities.topk(k=1, dim=1)
          training_accuracy_tensor = logits.view(*raw_labels.shape) == raw_labels.cuda()
          training_accuracy += torch.mean(training_accuracy_tensor.type(torch.FloatTensor))
          if (count % (int(len(trainloader)/10)+1) == 0):
              percent += 10
              print "%d %% Complete" % (percent)
      percent += 10
      print "%d %% Complete" % (percent)
      training_loss = running_loss / len(trainloader)
      # Calculate Accuracy
      training_accuracy = training_accuracy * 100 / len(trainloader)
      elapsed_time = time.time() - start_time
      print "Epoch %d / %d:" % (epoch+1, epochs)
      print "Time Elapsed = %d s" % (elapsed_time)
      print "Training Loss = %f" % (training_loss)
      print "Training Accuracy = %0.2f %%" % (training_accuracy)
      if ((epoch+1) % 10 == 0):
        print "Generate Reconstructions:"
        generate_reconstructions(reconstructions[:10].cpu(), images[:10].cpu(), epoch+1)
        print "Reconstructions Generated"
      overall_training_accuracy.append(training_accuracy)
      overall_training_loss.append(training_loss)
      history = {
      "train_loss": overall_training_loss,
      "train_accuracy": overall_training_accuracy
      }
  np.save("history.npy", history)
  torch.save(model, "model.pth")

elif mode == "Test":
  testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=100)
  loaded_model = torch.load("model.pth")
  loaded_model.cuda()
  loaded_model.eval()
  testing_accuracy = 0
  # Testing
  for images, labels in iter(testloader):
      class_probabilities, reconstructions = loaded_model(images.cuda())
      _, logits = class_probabilities.topk(k=1, dim=1)
      testing_accuracy_tensor = logits.view(*labels.shape) == labels.cuda()
      testing_accuracy += torch.mean(testing_accuracy_tensor.type(torch.FloatTensor))
  testing_accuracy = testing_accuracy * 100 / len(testloader)
  print "Test Accuracy = %0.2f %%" % (testing_accuracy)
  generate_reconstructions(reconstructions.cpu(), images.cpu(), "Test")
  plot_graph(testing_accuracy)

elif mode == "Pose":
  loaded_model = torch.load("model.pth")
  loaded_model.eval()
  test_indices = list(range(0, 11))
  test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
  testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=11)
  # Pose Relationship
  for images, _ in iter(testloader):
    _, reconstructions = loaded_model(images.cuda(), pose=True)
  generate_reconstructions(reconstructions.cpu())
