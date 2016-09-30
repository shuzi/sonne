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
        local input,input2,input3,input4
        if opt.selectfeat then
           input = train_x[begin]:cuda():narrow(1,1,opt.inputfeatures)
           input2 = train_x[begin]:cuda():narrow(1,7,opt.inputfeatures_G2)
           input3 = train_x[begin]:cuda():narrow(1,14,opt.inputfeatures_G3)
           input4 = train_x[begin]:cuda():narrow(1,19,opt.inputfeatures_G4)
        else
           input = train_x[begin]:cuda()
        end
        local inputsun = train_xsun[begin]
        local target = train_y[begin]
        

        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local f = 0
            if not opt.usesun then
               local output = model:forward(input)
               f = criterion:forward(output, target)
               loss = loss + f
               local df_do = criterion:backward(output, target)
               model:backward(input, df_do)
            else
               if opt.input2G then
                  local output = model:forward{input, input4, inputsun}
                  f = criterion:forward(output, target)
                  loss = loss + f
                  local df_do = criterion:backward(output, target)
                  model:backward({input, input2, inputsun}, df_do)
               elseif opt.input3G then
                  local output = model:forward{input, input2, input3, inputsun}
                  f = criterion:forward(output, target)
                  loss = loss + f
                  local df_do = criterion:backward(output, target)
                  model:backward({input, input2, input3, inputsun}, df_do)
               elseif opt.input4G then
                  local output = model:forward{input, input2, input3, input4, inputsun}
                  f = criterion:forward(output, target)
                  loss = loss + f
                  local df_do = criterion:backward(output, target)
                  model:backward({input, input2, input3, input4, inputsun}, df_do)
               else
                  local output = model:forward{input, inputsun}
                  f = criterion:forward(output, target)
                  loss = loss + f
                  local df_do = criterion:backward(output, target)
                  model:backward({input, inputsun}, df_do)
               end
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

function test(inputData, inputDataSun, inputTarget, state)
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
        local input,input2,input3,input4
        if opt.selectfeat then
            input = inputData[begin]:cuda():narrow(1,1,opt.inputfeatures)
            input2 = inputData[begin]:cuda():narrow(1,7,opt.inputfeatures_G2)
            input3 = inputData[begin]:cuda():narrow(1,14,opt.inputfeatures_G3)
            input4 = inputData[begin]:cuda():narrow(1,19,opt.inputfeatures_G4)
        else
            input = inputData[begin]:cuda()
        end
        local inputsun = inputDataSun[begin]
        local target = inputTarget[begin]
        local pred
        if not opt.usesun then
            pred = model:forward(input)
        else
            if opt.input2G then
                pred = model:forward{input, input4, inputsun}
            elseif opt.input3G then
                pred = model:forward{input, input2, input3, inputsun}
            elseif opt.input4G then
                pred = model:forward{input, input2, input3, input4, inputsun}
            else
                pred = model:forward{input, inputsun}
            end
        end
        f = loss:forward(pred, target)
        io.write(string.format("%s ", f))
        abs = abs + f
    end
    io.write(string.format("\n"))
    local currAbs = (abs / batches)/2000

    state.bestAbs = state.bestAbs or 10000000
    state.bestEpoch = state.bestEpoch or 0
    if currAbs < state.bestAbs then state.bestAbs = currAbs ; state.bestEpoch = epoch end
    print(string.format("Epoch %s Abs: %s, best Abs: %s on epoch %s at time %s", epoch, currAbs, state.bestAbs, state.bestEpoch, sys.toc() ))
end


