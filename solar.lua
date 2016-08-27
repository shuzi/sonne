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

cmd:text()
opt = cmd:parse(arg or {})


train_x = {}
train_y = {}
valid_x = {}
valid_y = {}
test_x = {}
test_y = {}

dofile("optim-msgd.lua")
dofile("loaddata.lua")

if opt.model == 1 then
  dofile("model_cnn_alexnet.lua")
elseif opt.model == 2 then
  dofile("model_cnn_alexnet_gru.lua")
elseif opt.model == 3 then 
  dofile("model_cnn_alexnet_rnn.lua")
elseif opt.model == 4 then 
  dofile("model_cnn_alexnet_lstm.lua")
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
       test(valid_x, valid_y, validState)
     end
     if opt.test then
       test(test_x, test_y, testState)
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

