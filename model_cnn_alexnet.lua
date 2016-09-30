model = nn.Sequential()
local am=opt.f1
model:add(nn.VolumetricConvolution(opt.inputfeatures, am, 1, 11, 11, 1, 4, 4, 0, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
model:add(nn.VolumetricConvolution(am, am*3, 1, 5, 5, 1, 1, 1, 0, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
model:add(nn.VolumetricConvolution(am*3, am*6, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricConvolution(am*6, am*6*0.75, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricConvolution(am*6*0.75, am*6*0.75, 1, 3, 3, 1, 1, 1, 0, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
model:add(nn.View(am*6*0.75*6*7*4))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(am*6*0.75*6*7*4, 1))


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

