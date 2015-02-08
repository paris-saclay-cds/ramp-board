------------------------------------------------------------------------
--[[ SaveReport ]]--
-- Interface, Observer
-- Simple logger that prints a report every epoch
------------------------------------------------------------------------
require 'torch'
require 'dp'
local SaveReport, parent = torch.class("dp.SaveReport", "dp.Logger")
SaveReport.isSaveReport = true

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function SaveReport:__init()
   parent.__init(self)
   self._reports = {}
end

function SaveReport:setup(config)
   parent.setup(self, config)
end

function SaveReport:doneEpoch(report)
   self._reports[report.epoch] = deepcopy(report)
end
