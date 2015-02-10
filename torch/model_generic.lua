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
csv2tensor = require 'csv2tensor'
require 'csvigo'
helpers  = require 'helpers'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generic torch model')
cmd:option('--train', 'train.csv', 'training CSV file (with targets as the first col)')
cmd:option('--test', 'test.csv', 'training CSV file (with no targets)')
cmd:option('--outputproba', 'proba.csv', 'test predictions proba CSV file')
cmd:option('--outputtargets', 'targets.csv', 'test predictions targets output CSV file')
cmd:text()
opt = cmd:parse(arg or {})

train_obj = helpers.get_data_and_column_names(opt.train)
test = csv2tensor.load(opt.test)

targets, proba = model(train_obj.X, train_obj.y, test)

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
