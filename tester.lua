--[[
Tester class to test models
By Valentin Radu @ University of Edinburgh
--]]

require 'optim'

local Tester = torch.class('Tester')

function Tester:__init()	
	self.name = 'Tester'	
end

function Tester:test(model, dataLoader, withCuda)
	-- This fuction evaluates model performances 

	local testAccuracy = testAccuracy or 0
	local testData = nil
	local testLabels = nil

	-- Confusion matrix 
	local confusion = optim.ConfusionMatrix(dataLoader.data['classes'])

	-- if self.onGPU then cutorch.synchronize() end
	local time = sys.clock()

	for i = 1, dataLoader:getNoTestBatches() do
		
		testData, testLabels = dataLoader:getTestBatch(i, testData, testLabels)

		if withCuda then
			testData:cuda()
			testLabels:cuda()
		end

		outputs = model:forward(testData)
		confusion:batchAdd(outputs, testLabels)

		-- disp progress
		xlua.progress(i * testData:size(1), dataLoader:getNoTestBatches() * testData:size(1))
	end

	time = sys.clock() - time
	time = time / dataLoader:getNoTestInstances()
	print("[*] Time to test one sample = " .. (time * 1000) .. 'ms')

	print(confusion)

	local meanF1 = 0
	local w_meanF1 = 0
	for t = 1, confusion.nclasses do
		precision = confusion.mat[t][t] / confusion.mat:select(2,t):sum() * 100
		recall = confusion.mat[t][t] / confusion.mat:select(1,t):sum() * 100
		f1 = 2 * precision * recall / (precision + recall)
		if (f1 ~= f1) then
			f1 = 0
		end
		print("class " .. t .. " -> F1: " .. f1)
		meanF1 = meanF1 + f1
		w_meanF1 = w_meanF1 + confusion.mat:select(1,t):sum() * f1

		--print('class[' .. confusion.classes[t] ..']\tprecision = ' .. precision .. '\trecall = ' .. recall .. '\tf1 = ' .. f1)
	end

	meanF1 = meanF1 / confusion.nclasses
	w_meanF1 = w_meanF1 / dataLoader:getNoTestInstances()

	print('Average F1-score: ' .. meanF1)
	print('Weighted F1-score: ' .. w_meanF1)

	testAccuracy = confusion.totalValid * 100

	collectgarbage()

	return testAccuracy, w_meanF1, confusion

end