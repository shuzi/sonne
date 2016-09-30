require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')
valid_x = torch.load("valid_x")
valid_xsun = torch.load("valid_xsun")
valid_y = torch.load("valid_y")

print(valid_x[1]:size())

print(valid_xsun[1])



function createConv(inputfeatures, am)
   local conv = nn.Sequential()
   conv:add(nn.VolumetricConvolution(inputfeatures, am, 2, 11, 11, 1, 4, 4, 0, 2, 2))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
   conv:add(nn.VolumetricConvolution(am, am*3, 2, 5, 5, 1, 1, 1, 0, 2, 2))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
   conv:add(nn.VolumetricConvolution(am*3, am*6, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricConvolution(am*6, am*6*0.75, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricConvolution(am*6*0.75, am*6*0.75, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
   conv:add(nn.View(conv:forward(valid_x[1]):storage():size()))
  -- conv:add(nn.View(am*6*0.75*6*7*4)
   return conv
end

model = nn.Sequential()
featextract = nn.ParallelTable()
featextract:add(createConv(36, 160))
featextract:add(nn.Linear(2,2))


model:add(featextract)
model = model:cuda()
print(model:forward{valid_x[1]:cuda(), valid_xsun[1]:cuda()})




