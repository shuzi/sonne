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
model:add(nn.View(720*6*7*4))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(720*6*7*4, 1))


if true then
  criterion = nn.MSECriterion()
elseif false then
  criterion = nn.AbsCriterion()
elseif false then
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

