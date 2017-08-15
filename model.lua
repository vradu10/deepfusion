--[[
Model class, facilitates interaction with model for parameters reset and initialization
By Valentin Radu @ University of Edinburgh
--]]

require 'nn'

local Model = torch.class('Model')

function Model:__init(model)
	self.name = 'Model'
	self.model = model
end

-- Initialize weights and bias with random small numbers
function Model:initializeParams(sigma)
	local sigma = sigma or 1
	local params, gradParams = self.model:getParameters()

	--params:normal():mul(sigma)
	params:uniform(-sigma, sigma)
	gradParams:zero()
end

function Model:initializeParams2()
	for i,m in ipairs(self.model.modules) do
		if torch.typename(m) == "nn.Linear" then
			m.weight:zero()
			
			inSize = m.weight:size(2)
			outSize = m.weight:size(1)
			B = math.sqrt(6) / math.sqrt(inSize + outSize)

			m.weight:copy(torch.rand(inSize * outSize))
			m.weight:mul(2 * B):add(-B)

			m.bias:zero()
		end
	end

end

-- Clear any input/output and gradient data
function Model:clearSequential()
	for i, m in ipairs(self.model.modules) do
		if m.output then m.output = torch.Tensor() end
		if m.gradInput then m.gradInput = torch.Tensor() end
		if m.gradWeight then m.gradWeight = torch.Tensor() end
		if m.gradBias then m.gradBias = torch.Tensor() end
	end
	collectgarbage()
	return model
end
