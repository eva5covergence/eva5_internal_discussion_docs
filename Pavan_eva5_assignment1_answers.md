**1. What are Channels and Kernels (according to EVA)?**


*Channels:*

Channels are also called as feature maps (feature map name itself suggests that it is a collection of features).
Convolution operation over input by a kernel/filter gives a single channel as output. So if there are n filters convolve over input will get n channels as output. (We apply some non-linearity function like ReLU on top of it)

So Channel is a representation of an image which has features (like edges/ gradients/ patterns/ textures / objects).


*Kernels:*

Kernel is also called as filter or feature extractor.

In conventional image processing techniques Kernel or filters tailored manually but where as in CNN, network learns automatically through optimization algorithms (gradient descent) in the back propagation step.

Example of conventional image processing filter is - Sobel filter which helps to find out the edges of the input image by calculating x-gradient and y-gradient & find the resulting gradient’s magnitude from them and it’s direction (angle - theta).

Kernel/Filter convolve over input (image or representation of image [hidden layers activations]) and gives the output channels/feature_maps.


**2. Why should we (nearly) always use 3x3 kernels?**



When the kernel size as small as possible leads to learn more information from the input and captures the spatial relationships from the neighbouring pixels.

Let us say there is an image of size 100 x 100 which has many small objects inside the image, if you apply a large size kernel say 20x20, it misses the information of these small objects. But if you try with smaller kernel size it will capture the information of small objects as well as it convolves over a small region of the image again and again.

And also when we chose large kernel size, each convolution operation has lot of common region (common receptive field) where it is convolving when stride is also less. So many duplicate operations.

So you may question that why can’t we choose 1x1 or 2x2?

When we do convolution operation by a kernel over an image, the image size(height and width) reduces gradually unless we do padding on the input. Here reducing means we are throwing out some information, so with padding we can maintain same size as input with 3x3 but can’t with 2x2.

*Example:*

Output of convolution formulated as (n-k+2p)/s +1

n - input dimension
k - kernel size
p - padding
s - stride 

Let us say we have 10x10 input, 2x2 filter, padding=1, stride=1

Output dimension - (10-2+2)/1 + 1 = 11. Here we got 11 but we need 10 which is size of our input

Let us say we have 10x10 input, 3x3 filter, padding=1, stride=1

Output dimension - (10-3+2)/1 + 1 = 10. Here we got same size as input


And coming to 1x1, it is mostly used by pre-trained network authors (ex: GoogleNet) to control the number of output channels/feature_maps.

Ex:

Input is 10x10x200, we can reduce 10x10x200 to 10x10x1 by applying a single 1x1 filter
Incase input is 10x10x1 and you apply one 1x1 filter on top of it, you will get 10x10x1 itself. It is a kind of non-linearity effect on the input.

Here 1x1 kernel does not learn the spatial relationship with the neighbouring pixels.

**3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)**

99 convolution operations need to performed with 3x3 over 199x199 to reach 1x1

199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 > 179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 > 159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 > 139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 > 119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 > 99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 > 79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 > 59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 > 39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 > 19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1


**4. How are kernels initialised? **

I referred below article to understand journey of weight initialisation in DNN. I tried all below experiments put my understandings here.

(https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79#:~:text=The%20aim%20of%20weight%20initialization,through%20a%20deep%20neural%20network.)


In DNN/CNN, network performs many consecutive matrix multiplications from layer1 till the last layer. And we have choice of various activation functions to choose for non-linearity over linear outputs (logits).

So initialising the weights properly is a crucial step for model learning to converge well. Our aim of weight initialisation such that each layer’s activation outputs with mean around zero and standard deviation around 1.

Naive ideas like initialise the weights with zeros or normally distributed random values does not work.

For example, we simulate 100 layer network by multiplying input “x” with weight matrix “a” 100 times

***Experiment1:*** Initialise weights with normalised random values

x= torch.randn(512,512)
for i in range(100):
  try:
    a = torch.randn(512,512)
    x = a @ x

x.mean(), x.std()

(tensor(nan), tensor(nan))

Above tensor values are nan because they exploded with very big values.

***Experiment2:*** Initialise weights with normalised random values and make them smaller by multiplying with 0.01

x = torch.randn(512)

for i in range(100):
  a = torch.randn(512,512) * 0.01
  x = a @ x

x.mean(), x.std() ### Vanishing gradients

(tensor(0.), tensor(0.))

Above tensor values are zeros. That means the output got vanished 


Above Both experiments failed. Experiment1 failed because the outputs got exploded (I observed it exploded at 29th multiplication or 29th layer) and Experiment2 failed because the outputs got vanished.

The root cause of experiment1’s failure is that every layer outputs’ variance is around 512 in each layer, so got exploded. Interestingly one observation here is that our input x size is also 512.


***Experiment3:*** Initialise the weights with normalised random values and multiply with square_root(1/ (num_of_inputs)) based on the above learning.

x = torch.randn(512)
for i in range(100):
  try:
    a = torch.randn(512,512)*math.sqrt(1./512)
    x = a @ x
  except:
    print(i)

x.mean(), x.std()

(tensor(0.0429), tensor(0.8416))

And we got valid results after 100 layers multiplications. And I observed the mean of each layer is around 0 and standard deviation is 1

So Experiment3 approach helped to maintain mean of 0 and SD 1, leads no issues (exploding or vanishing outputs)


Note: We didn’t use any non-linear functions in experiment1 or experiment2 or experiment 3. But in real time non-linear function need to be applied to classify or regress well on real time data as most of the real time data is non-linear in nature.

***Experiment4:*** Let’s use non-linear activation function on top of experiment3

x = torch.randn(512)
for i in range(100):
  try:
    a = torch.randn(512,512)*math.sqrt(1./512)
    x = tanh(a @ x)
  except:
    print(i)

x.mean(), x.std()

(tensor(0.0009), tensor(0.0549))

And we got decent results after 100 layers multiplications.

Experiment5: Based on Experiment4, Xavier Glorot & Yoshua Bengio found below little modified approach of experiment 3 empirically 

Initialise the weights with random uniformed distribution that’s bounded between +/-(sqrt(6/[num_inputs+num+outputs]))

def xavier(m,h):
  return torch.Tensor(m,h).uniform_(-1,1)*math.sqrt(6./(m+h))

x = torch.randn(512)
for i in range(100):
    a = xavier(512,512)
    x = tanh(a @ x)

x.mean(), x.std()

(tensor(0.0024), tensor(0.0794))

Experiment5 which found by Xavier & Yoshua gave good results. And it maintained around mean of 0 and SD of 1 at each and every layer.

But there is new problem came when we change the activation function from Tanh to ReLU. ReLU came into picture and widely used because of slow learning or no learning at the extreme values of Tanh or sigmoid or soft sign functions, where the slope/gradient is almost near to zero. So weights won’t adjust much. So no learning.

When we use ReLU activation function with Xavier weight initialisation, we got an issue again. Let us in Experiment 5 below

***Experiment 5:*** Xavier weight initialisation and tanh as activation function.

def relu(x): return x.clamp_min_(0.)

x = torch.randn(512)
for i in range(100):
    a = xavier(512,512)
    x = relu(a @ x)

x.mean(), x.std()
(tensor(3.7465e-16), tensor(5.6924e-16))

Experiment 5 got failed as you could see above outputs got vanished. 

And below I observed that the standard deviation of each layer is around 16 (which is sqrt(512/2)) (Note: our number of inputs is 512) when we used ReLU with simple random standard normalisation initialised weights.

mean, var = 0.0, 0.0

for i in range(10000):
  x = torch.randn(512)
  a = torch.randn(512,512)
  y = relu(a @ x)
  mean += y.mean().item()
  var += y.pow(2).mean().item()

mean/10000, math.sqrt(var/10000)
(9.01886348772049, 15.995159460657934)

So we can resolve this issue if we multiply randomly normalised weights with sort(2/512) which gives the results of each layers outputs having mean of zero and standard deviation as 1.

***Experiment6:*** From the above reasoning that standard deviation of activation outputs for each layer is around sqrt(512/2) - (He initialisation)

Above logic formulated by Kaiming He and we mostly use He initialisation in CNN/DNN whenever we use ReLU activation function

def kaiming(m,n):
  return torch.randn(m,n)*math.sqrt(2./m)

x = torch.randn(512)
for i in range(100):
    a = kaiming(512,512)
    x = relu(a @ x)

x.mean(), x.std()

(tensor(0.4737), tensor(0.6989))

We got valid outputs even with ReLU activation function when we used He Initialisation of weights.

So our weight initialisation aim is to have mean of 0 and standard deviation 1 in each and every layer’s outputs, so that our computation even in deeper networks will remain go smoothly.

We need to define our own customised weight initialisation when we try to use a customised activation function by keeping above aim in mind.

**5. What happens during the training of a DNN?**

During the training of DNN, it is trying to find a function in a mathematical space such that f(input) = output. This function has lot of parameters or weights. Network will find out these weights through optimisation algorithm like gradient descent applies on loss function like L(f(input),output)

In detail, following things happen in DNN

*Feed forward:*

1. Define network architecture (layers and neurones in each layer, activation function)
2. Initialise the weights
3. Pass the input through root node of network (computation graph) and get the output
4. Calculate the loss using appropriate loss function L(f(input),output)

*Backward propagation:*

1. Calculate the gradients or partial derivatives - Change in loss w.r.to change in weights
2. Adjust the weights using respective gradients w.r.to those weights - say simple vanilla SGD ==>  W -= (learning_rate) * (dL/dW)

So here one iteration(feed forward and backward propagation) happen per batch of inputs and these iterations happen till completion of all the inputs. And we will call it as an epoch.

And multiple epochs can be run till model fits the data perfectly without getting overfit.

The weights/kernals/filters/feature extractors which learnt during the training of DNN extract features in every layer which helps to find out the output as accurately as possible.


