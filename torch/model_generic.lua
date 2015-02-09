require 'torch'
require 'model'
csv2tensor = require 'csv2tensor'
require 'csvigo'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generic torch model')
cmd:option('--train', 'train.csv', 'training CSV file (with targets as the first col)')
cmd:option('--test', 'test.csv', 'training CSV file (with no targets)')
cmd:option('--output', 'pred.csv', 'test predictions output CSV file')
cmd:text()
opt = cmd:parse(arg or {})

train = csv2tensor.load(opt.train)
test = csv2tensor.load(opt.test)

pred = torch.Tensor.totable(torch.gt(test, 0.3))
for i, k in ipairs(pred) do
    pred[i] = {tostring(pred[i])}
end

csvigo.save{path=opt.output, data=pred, header=false}
