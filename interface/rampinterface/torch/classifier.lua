require 'torch'


function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

function argmax_2D(matrix)
   local nRows = matrix:size(1)
   local result = torch.Tensor(nRows)
   for i = 1, nRows do
      result[i] = argmax_1D(matrix[i])
   end
   return result
end


require 'model'
require 'csvigo'
helpers  = require 'helpers'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generic torch model')
cmd:option('--X', 'dataset', 'training CSV file (with targets as the first col)')
cmd:option('--y', 'labels', 'training CSV file (with no targets)')
cmd:option('--train', 'labels', 'training CSV file (with no targets)')
cmd:option('--test', 'labels', 'training CSV file (with no targets)')
cmd:option('--outputproba', 'proba.csv', 'test predictions proba CSV file')
cmd:option('--outputtargets', 'targets.csv', 'test predictions targets output CSV file')
cmd:text()
opt = cmd:parse(arg or {})


X = torch.load(opt.X)
print(X:size())
y = torch.Tensor(helpers.get_indexes(opt.y))
print(y:size())

train = torch.LongTensor(helpers.get_indexes(opt.train))
valid = torch.LongTensor(helpers.get_indexes(opt.test))

X_train = X:index(1, train)
X_valid = X:index(1, valid)
y_train = y:index(1, train)
y_valid = y:index(1, valid)

print(X_train:size())
print(X_valid:size())
print(y_train:size())
print(y_valid:size())

targets, proba = model(X_train, y_train, X_valid)

targets = torch.Tensor.totable(targets)
proba = torch.Tensor.totable(proba)

for i, k in ipairs(targets) do
    targets[i] = {tostring(targets[i])}
end

for i, k in ipairs(proba) do
    for j, l in ipairs(proba[i]) do
        proba[i][j] = tostring((proba[i][j]))
    end
end

csvigo.save{path=opt.outputproba, data=proba, header=false}
csvigo.save{path=opt.outputtargets, data=targets, header=false}
