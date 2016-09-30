
model = nn.Sequential()
model:add(nn.VolumetricConvolution(10, 1000, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 3, 3))
model:add(nn.VolumetricConvolution(1000, 1000, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 3, 3))
model:add(nn.View(1000*6*28*19))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1000*6*28*19, 1))



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

