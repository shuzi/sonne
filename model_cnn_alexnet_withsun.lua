len=0
function createConv(inputfeatures, am, kt1, kt2, kt3, kt4, kt5)
   local conv = nn.Sequential()
   conv:add(nn.VolumetricConvolution(inputfeatures, am, kt1, 11, 11, 1, 4, 4, 0, 2, 2))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
   conv:add(nn.VolumetricConvolution(am, am*3, kt2, 5, 5, 1, 1, 1, 0, 2, 2))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
   conv:add(nn.VolumetricConvolution(am*3, am*6, kt3, 3, 3, 1, 1, 1, 0, 1, 1))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricConvolution(am*6, am*6*0.75, kt4, 3, 3, 1, 1, 1, 0, 1, 1))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricConvolution(am*6*0.75, am*6*0.75, kt5, 3, 3, 1, 1, 1, 0, 1, 1))
   conv:add(nn.ReLU(true))
   conv:add(nn.VolumetricMaxPooling(1, 3, 3, 1, 2, 2))
   len = conv:forward(train_x[1]:narrow(1,1,opt.inputfeatures)):storage():size()
   conv:add(nn.View(len))
 
--   conv:add(nn.View(am*6*0.75*6*7*4))
   return conv
end


model = nn.Sequential()
featextract = nn.ParallelTable()
featextract:add(createConv(opt.inputfeatures, opt.f1, opt.kt1, opt.kt2, opt.kt3, opt.kt4, opt.kt5))
featextract:add(nn.Linear(2,2))
model:add(featextract)
model:add(nn.JoinTable(1))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(len+2, 1))


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

