
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')
test_x = torch.load("test_x")




model = nn.Sequential()
model:add(nn.VolumetricConvolution(10, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.3))
model:add(nn.VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))

model:add(nn.VolumetricConvolution(64, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))


model:add(nn.VolumetricConvolution(128, 256, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(256, 256, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(256, 256, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))


model:add(nn.VolumetricConvolution(256, 512, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(512, 512, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(512, 512, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))


model:add(nn.VolumetricConvolution(512, 512, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(512, 512, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.4))
model:add(nn.VolumetricConvolution(512, 512, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))



model:add(nn.View(512*6*8*5))
model:add(nn.Dropout(0.5))
--model:add(nn.Linear(512*6*8*5, 512*6*8*5))
--model:add(nn.ReLU(true))
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(512*6*8*5,1))



if opt.lossmode == 1 then
  criterion = nn.MSECriterion()
elseif opt.lossmode == 2 then
  criterion = nn.AbsCriterion()
elseif opt.lossmode == 3 then
  MSE = nn.MSECriterion():cuda()
  ABS = nn.AbsCriterion():cuda()
  criterion = nn.MultiCriterion():add(MSE, opt.criterionMSEWeight):add(ABS)
else
end

for k,v in pairs(model:findModules('nn.VolumetricConvolution')) do
      local n = v.kW*v.kH*v.kT*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
end


model:cuda()
criterion:cuda()
if model then
   parameters,gradParameters = model:getParameters()
   print("Model Size: ", parameters:size()[1])
   parametersClone = parameters:clone()
end
print(model)
print(criterion)


