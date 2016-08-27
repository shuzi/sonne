model = nn.Sequential()
model:add(nn.VolumetricConvolution(10, 160, 1, 11, 11, 1, 4, 4, 0, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
model:add(nn.VolumetricConvolution(160, 480, 1, 5, 5, 1, 1, 1, 0, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
model:add(nn.VolumetricConvolution(480, 960, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricConvolution(960, 720, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricConvolution(720, 720, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
model:add(nn.Transpose({1,2}))
model:add(nn.View(1, 6, 720*7*4))
model:add(cudnn.RNNReLU(720*7*4, 5000, 1, true))
model:add(nn.Max(2))
--model:add(nn.Select(2,-1))
model:add(nn.ReLU())
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(5000, 1))


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

model:cuda()
criterion:cuda()
if model then
   parameters,gradParameters = model:getParameters()
   print("Model Size: ", parameters:size()[1])
   parametersClone = parameters:clone()
end
print(model)
print(criterion)

