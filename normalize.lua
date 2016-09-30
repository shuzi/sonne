
dd = npy4th.loadnpy('/DT/SPEED_CROSS.npy')
y = {}
for i=1,dd:size(1) do y[dd[i][1]] = dd[i][2] end

dir = opt.path .. 'train/'
n=1
for k,v in pairs(paths.dir(dir)) do
        if v == '.' then goto continue end
        if v == '..' then goto continue end
        ts = tonumber(v:match('%d+'))
        train_x[n] = npy4th.loadnpy(dir .. v)
        train_x[n] = train_x[n]:transpose(1,2):contiguous():float()
        train_y[n] = torch.Tensor{y[ts]}:cuda()
        print( train_y[n])
        n = n+1
        ::continue::
end

dir = opt.path .. 'test/'
n=1
for k,v in pairs(paths.dir(dir)) do
        if v == '.' then goto continue end
        if v == '..' then goto continue end
        ts = tonumber(v:match('%d+'))
        test_x[n] = npy4th.loadnpy(dir .. v)
        test_x[n] = test_x[n]:transpose(1,2):contiguous():float()
        test_y[n] = torch.Tensor{y[ts]}:cuda()
        n = n+1
        ::continue::
end

dir = opt.path .. 'valid/'
n=1
for k,v in pairs(paths.dir(dir)) do
        if v == '.' then goto continue end
        if v == '..' then goto continue end
        ts = tonumber(v:match('%d+'))
        valid_x[n] = npy4th.loadnpy(dir .. v)
        valid_x[n] = valid_x[n]:transpose(1,2):contiguous():float()
        valid_y[n] = torch.Tensor{y[ts]}:cuda()
        n = n+1
        ::continue::
end

print("Data loading finished")
print("train size:", #train_x, #train_y)
print("valid size:", #valid_x, #valid_y)
print("test size:", #test_x, #test_y)

