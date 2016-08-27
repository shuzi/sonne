optimState = {
      lr = opt.learningRate,
      lrd = opt.learningRateDecayFactor,
      lrp = opt.learningRateDecayPower,
      mom = opt.momentum
}
optimMethod = optim.msgd

function train()
    epoch = epoch or 1
    local time = sys.clock()
    model:training()
    local batches = #train_x/opt.batchSize
    local bs = opt.batchSize
    shuffle = torch.randperm(batches)
    local loss=0
    for t = 1,batches,1 do
        local begin = (shuffle[t] - 1)*bs + 1
        local input = train_x[begin]:cuda()
        local target = train_y[begin]

        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local f = 0
            if true then
               local output = model:forward(input)
               f = criterion:forward(output, target)
               loss = loss + f
               local df_do = criterion:backward(output, target)
               model:backward(input, df_do)
            else
               local output = model:forward(input)
               f = criterion:forward(output, target)
               local df_do = criterion:backward(output, target)
               model:backward(input, df_do)
            end
            if opt.L1reg ~= 0 then
               local norm, sign = torch.norm, torch.sign
               f = f + opt.L1reg * norm(parameters,1)
               gradParameters:add( sign(parameters):mul(opt.L1reg) )
            end
            if opt.L2reg ~= 0 then
               parametersClone:copy(parameters)
               gradParameters:add( parametersClone:mul(opt.L2reg) )
            end
            gradParameters:clamp(-opt.gradClip, opt.gradClip)
            return f,gradParameters
        end
        optimMethod(feval, parameters, optimState)
    end

    time = sys.clock() - time
    print("\n==> time for 1 epoch = " .. (time) .. ' seconds' .. " avg. training loss is: " .. string.format("%.3f", loss/batches) )
end

function test(inputData, inputTarget, state)
    local time = sys.clock()
    model:evaluate()
    loss = nn.AbsCriterion()
    local bs = opt.batchSize
    local batches = #inputData/bs
    local curr = -1
    local abs=0
    for t = 1,batches,1 do
        curr = t
        local begin = (t - 1)*bs + 1
        local input = inputData[begin]
        local target = inputTarget[begin]
        local pred = model:forward(input)
        f = loss:forward(pred, target)
        abs = abs + f
    end
    local currAbs = (abs / batches)/2000

    state.bestAbs = state.bestAbs or 10000000
    state.bestEpoch = state.bestEpoch or 0
    if currAbs < state.bestAbs then state.bestAbs = currAbs ; state.bestEpoch = epoch end
    print(string.format("Epoch %s Abs: %s, best Abs: %s on epoch %s at time %s", epoch, currAbs, state.bestAbs, state.bestEpoch, sys.toc() ))

end


