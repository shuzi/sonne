npy4th = require 'npy4th'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
cmd = torch.CmdLine('_')
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'number of threads')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-learningRateDecayFactor', 0, 'learning rate decaying factor')
cmd:option('-learningRateDecayPower', 1, 'learning rate decaying power')
cmd:option('-momentum', 0, 'momentum for msgd')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-path', "/raid/rui/DT/", 'path for train/valid/test')
cmd:option('-criterionMSEWeight', 1, "weight of MSE in loss function")
cmd:option('-epoch', 200, 'maximum epoch')
cmd:option('-L1reg', 0, 'L1 regularization coefficient')
cmd:option('-L2reg', 1e-4, 'L2 regularization coefficient')
cmd:option('-valid', false, 'run valid')
cmd:option('-test', false, 'run test')
cmd:option('-outputprefix', 'none', 'output file prefix')
cmd:option('-gradClip', 0.5, 'gradient clamp')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-dropout', 0, 'dropout probability')
cmd:option('-lossmode', 1, 'loss mode 1 2 3')
cmd:option('-model', 1, 'select model')
cmd:option('-savedata', false, 'save data tensor')
cmd:option('-loaddata', false, 'load data tensor')
cmd:option('-normdata', false, 'normalize data tensor')
cmd:option('-combvalid', false, 'combine train/valid data tensor')
cmd:option('-selectfeat', false, 'use first 10 features')
cmd:option('-inputfeatures', 36, 'amount of input features')
cmd:option('-inputfeatures_G2', 7, 'amount of input features')
cmd:option('-inputfeatures_G3', 5, 'amount of input features')
cmd:option('-inputfeatures_G4', 6, 'amount of input features')
cmd:option('-f1', 160, 'amount of first layer output features')
cmd:option('-f1_G2', 160, 'amount of first layer output features')
cmd:option('-f1_G3', 160, 'amount of first layer output features')
cmd:option('-f1_G4', 160, 'amount of first layer output features')
cmd:option('-usesun', false, 'use sun features')
cmd:option('-kt1', 1, 'kt for conv layer 1')
cmd:option('-kt2', 1, 'kt for conv layer 2')
cmd:option('-kt3', 1, 'kt for conv layer 3')
cmd:option('-kt4', 1, 'kt for conv layer 4')
cmd:option('-kt5', 1, 'kt for conv layer 5')
cmd:option('-kt1_G2', 1, 'kt for conv layer 1')
cmd:option('-kt2_G2', 1, 'kt for conv layer 2')
cmd:option('-kt3_G2', 1, 'kt for conv layer 3')
cmd:option('-kt4_G2', 1, 'kt for conv layer 4')
cmd:option('-kt5_G2', 1, 'kt for conv layer 5')
cmd:option('-input2G', false, 'two groups of input')
cmd:option('-input3G', false, 'three groups of input')
cmd:option('-input4G', false, 'four groups of input')


cmd:text()
opt = cmd:parse(arg or {})

train_x = {}
train_xsun = {}
train_y = {}
valid_x = {}
valid_xsun = {}
valid_y = {}
test_x = {}
test_xsun = {}
test_y = {}

torch.setdefaulttensortype('torch.FloatTensor')

dofile("optim-msgd.lua")
if opt.loaddata then
   train_x = torch.load("train_x")
   train_xsun = torch.load("train_xsun")
   train_y = torch.load("train_y")
   valid_x = torch.load("valid_x")
   valid_xsun = torch.load("valid_xsun")
   valid_y = torch.load("valid_y")
   test_x = torch.load("test_x")
   test_xsun = torch.load("test_xsun")
   test_y = torch.load("test_y")
else
   dofile("loaddata.lua")
end

if opt.savedata then
   torch.save("train_x", train_x)
   torch.save("train_xsun", train_xsun)
   torch.save("train_y", train_y)
   torch.save("valid_x", valid_x)
   torch.save("valid_xsun", valid_xsun)
   torch.save("valid_y", valid_y)
   torch.save("test_x", test_x)
   torch.save("test_xsun", test_xsun)
   torch.save("test_y", test_y)
end

if opt.normdata then
   mean = torch.load("mean")
   std = torch.load("std")
   for i=1,#train_x do
      for j =1,10 do
         train_x[i][j]:add(-mean[j])
         train_x[i][j]:div(std[j])
      end
   end
   for i=1,#valid_x do
      for j =1,10 do
         valid_x[i][j]:add(-mean[j])
         valid_x[i][j]:div(std[j])
      end
   end
   for i=1,#test_x do
      for j =1,10 do
         test_x[i][j]:add(-mean[j])
         test_x[i][j]:div(std[j])
      end
   end 
end

if opt.combvalid then
   for i=1,#valid_x do
        train_x[#train_x+1] = valid_x[i]
        train_xsun[#train_xsun+1] = valid_xsun[i]
        train_y[#train_y+1] = valid_y[i]
   end
   valid_x = nil
   valid_xsun = nil
   valid_y = nil
end



if opt.model == 1 then
  dofile("model_cnn_alexnet.lua")
elseif opt.model == 2 then
  dofile("model_cnn_alexnet_gru.lua")
elseif opt.model == 3 then 
  dofile("model_cnn_alexnet_rnn.lua")
elseif opt.model == 4 then 
  dofile("model_cnn_alexnet_lstm.lua")
elseif opt.model == 5 then
  dofile("model_cnn_alexnet2.lua")
elseif opt.model == 6 then
  dofile('model_cnn_vgg.lua')
elseif opt.model == 7 then
  dofile('model_cnn.lua')
elseif opt.model == 8 then
  dofile('model_cnn_alexnet_withsun.lua')
elseif opt.model == 9 then
  dofile('model_cnn_alexnet_gru_withsun.lua')
elseif opt.model == 10 then
  dofile('model_cnn_alexnet_withsun_2G.lua')
end


dofile("train.lua")
collectgarbage()
collectgarbage()
torch.manualSeed(opt.seed)
math.randomseed(opt.seed)

sys.tic()
epoch = 1
validState = {}
testState = {}
while epoch <= opt.epoch do
   train()
   if true then
     if opt.valid then
       test(valid_x, valid_xsun, valid_y, validState)
     end
     if opt.test then
       test(test_x, test_xsun, test_y, testState)
     end
   end

   if opt.outputprefix ~= 'none' then
      if opt.saveMode == 'last' and epoch == opt.epoch then
         local t = sys.toc()
         saveModel(t + opt.prevtime)
         local obj = {
            em = model:get(1).weight,
            s2i = mapWordStr2WordIdx,
            i2s = mapWordIdx2WordStr
         }
         torch.save(opt.outputprefix .. string.format("_%010.2f_embedding", t + opt.prevtime), obj)
      elseif opt.saveMode == 'every'  then
         local t = sys.toc()
         saveModel(t + opt.prevtime)
         local obj = {
            em = model:get(1).weight,
            s2i = mapWordStr2WordIdx,
            i2s = mapWordIdx2WordStr
         }
         torch.save(opt.outputprefix .. string.format("_%010.2f_embedding", t + opt.prevtime), obj)
      end
   end
   epoch = epoch + 1
end

