
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')
train_x = torch.load("train_x")
valid_x = torch.load("valid_x")
test_x = torch.load("test_x")

train_x_sum = nn.CAddTable():forward(train_x)/#train_x
valid_x_sum = nn.CAddTable():forward(valid_x)/#valid_x
test_x_sum = nn.CAddTable():forward(test_x)/#test_x

train_x_sum = (train_x_sum + valid_x_sum + test_x_sum)/3 
print(train_x_sum:size())
print(train_x[1]:size())

mean = {}
for i=1,10 do
  mean[i] = train_x_sum[i]:sum()/(train_x_sum:size(2) * train_x_sum:size(3) * train_x_sum:size(4))
end

std={0,0,0,0,0,0,0,0,0,0}
for i=1,#train_x do
  for j=1,10 do
     train_x[i][j] = torch.pow(train_x[i][j] - mean[j], 2)
     std[j] = std[j]+ train_x[i][j]:sum()
  end
end
for i=1,#valid_x do
  for j=1,10 do
     valid_x[i][j] = torch.pow(valid_x[i][j] - mean[j], 2)
     std[j] = std[j]+ valid_x[i][j]:sum()
  end
end
for i=1,#test_x do
  for j=1,10 do
     test_x[i][j] = torch.pow(test_x[i][j] - mean[j], 2)
     std[j] = std[j]+ test_x[i][j]:sum()
  end
end



for i=1,10 do
  std[i] = std[i]/(  (#train_x + #valid_x + #test_x)*train_x_sum:size(2) * train_x_sum:size(3) * train_x_sum:size(4) - 1 )
  std[i] = math.sqrt(std[i])
end

print(mean)
print(std)
torch.save("mean", mean)
torch.save("std", std)


