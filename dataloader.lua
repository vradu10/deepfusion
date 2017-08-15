--[[
Data class to prepare data for training and evaluation
By Valentin Radu @ University of Edinburgh
--]]

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(config)
	self.name = 'DataLoader'

	self.config = config

	self:prepareData()	-- populates self.data
	self:applyDataDensity()

	print("Data ready:")
	print("- train " .. self.data['trainData']:size(1) .. " x " .. self.data['trainData']:size(2) .. 
				" & " .. self.data['trainLabel']:size(1) .. " x 1")
	print("- test   " .. self.data['testData']:size(1) .. " x " .. self.data['testData']:size(2) .. 
				" & " .. self.data['testLabel']:size(1) .. " x 1")
end

function DataLoader:prepareData()
	local fileName = self.config.path
	if self.config.validation_mode == 'cross-validation' then
		fileName = fileName .. self.config.preprocess .. "_fold_" .. self.config.current_fold .. ".t7b"
	elseif self.config.validation_mode == 'train-test' then
		fileName = fileName .. self.config.preprocess .. "_tt_" .. self.config.trainFile .. ".t7b"
	else
		-- when using leave-one-out evaluation method
		fileName = fileName .. self.config.preprocess .. "_" .. self.config.trainFile .. ".t7b"
	end

	-- check if file exists
	local file = io.open(fileName, "r")
	if file ~= nil then 
		io.close(file) 
		self.data = torch.load(fileName)
		print("Loading data...")
	else
		-- parse data and create fast load file
		require 'fileparser'
		
		parser = FileParser(self.config)
		self.data = parser:parseDataFile()

--		parser:generateARFF(1)
--		parser:generateARFF(2)
		
		torch.save(fileName, self.data)
		print("Created and Stored data")
	end
end

-- returns the next batch in the training data in order with an internal iterator.
-- When the internal iterator reaches the end it repopulates a new distribution.
function DataLoader:getTrainBatch(inputs, labels)
	local inputs = inputs or torch.Tensor(self.config.batch_size, self.config.input_size)
	local labels = labels or torch.Tensor(self.config.batch_size)

	if not self.config.current_batch or self.config.current_batch == 0 then
		local no_batches = self:getNoTrainBatches()
		self:splitTrainBarches(self.config.batch_size, no_batches)
		self.config.current_batch = no_batches
		--print("Batches Reset\n")
	end

	-- get current split indices
	shuffled = self.indices[self.config.current_batch]
	self.config.current_batch = self.config.current_batch - 1

	-- form batch
	for i = 1, inputs:size(1) do
		inputs[i]:copy(self.data['trainData'][shuffled[i]])
		labels[i] = self.data['trainLabel'][shuffled[i]]
	end

	return inputs, labels
end

function DataLoader:getTestBatch(batch_index, input, labels)
	local inputs = inputs or torch.Tensor(self.config.batch_size, self.config.input_size)
	local labels = labels or torch.Tensor(self.config.batch_size)

	if batch_index < 1 or batch_index > self:getNoTestBatches() then
		return nil, nil
	end

	-- if batch_index == self:getNoTestBatches() then
	-- 	local smaller_batch_size = self.data['testData']:size(1) - 
	-- 							self.config.batch_size * (self:getNoTestBatches() - 1)
	-- 	inputs = torch.Tensor(smaller_batch_size, self.config.input_size)
	-- 	labels = torch.Tensor(smaller_batch_size)
	-- end

	for i = 1, inputs:size(1) do
		dataset_index = self.config.batch_size * (batch_index - 1) + i
		inputs[i]:copy(self.data['testData'][dataset_index])
		labels[i] = self.data['testLabel'][dataset_index]
	end

	return inputs, labels

end

function DataLoader:splitTrainBarches(batch_size, no_batches)
	-- split from training data
	self.indices = torch.randperm(no_batches * batch_size):split(batch_size)

	-- split a fair distribution of classes across batches
	-- -- count the numnber of classes	
	-- no_classes = #self.data['classes']
	-- no_classes_size = {}
	-- indices = {}
	-- for i = 1, no_classes do
	-- 	no_classes_size[i] = 0
	-- 	indices[i] = {}
	-- end

	-- for i = 1, #self.data['trainData']:size() do
	-- 	class = self.data['trainLabel'][i]
	-- 	no_classes_size[class] = no_classes_size[class] + 1
	-- 	indices[class][#indices[class] + 1] = i
	-- end

	-- -- fair distribution
	-- fair_per_batch = {}
	-- for i = 1, no_classes do
	-- 	fair_per_batch = torch.floor(no_classes_size[i] / batch_size)
	-- end

	-- -- distribute indices across batches TODO
end

function DataLoader:applyDataDensity()
	if self.config.dataset_fraction >= 1 then
		return
	end

	print("data size before density: " .. self.data['trainData']:size(1) .. 
			" and " .. self.data['testData']:size(1))

	local keep_train_size = torch.round(self.config.dataset_fraction * self.data['trainData']:size(1))
	local keep_test_size = torch.round(self.config.dataset_fraction * self.data['testData']:size(1))

	self.data['trainData'] = self.data['trainData']:narrow(1,1,keep_train_size):clone()
	self.data['trainLabel'] = self.data['trainLabel']:narrow(1,1,keep_train_size):clone()

	self.data['testData'] = self.data['testData']:narrow(1,1,keep_test_size):clone()
	self.data['testLabel'] = self.data['testLabel']:narrow(1,1,keep_test_size):clone()

	print("data size after density: " .. self.data['trainData']:size(1) .. 
			" and " .. self.data['testData']:size(1))	
	collectgarbage()
end

function DataLoader:resetTrainBatches()
	self.config.current_batch = 0
end

function DataLoader:getNoTrainInstances()
	return self.data['trainData']:size(1)
end

function DataLoader:getNoTrainBatches()
	-- cut off the last instances not forming a batch
	return torch.floor(self.data['trainData']:size(1) / self.config.batch_size)	
end

function DataLoader:getNoTestBatches()
	return torch.floor(self.data['testData']:size(1) / self.config.batch_size)
end

function DataLoader:getNoTestInstances()
	return self.data['testData']:size(1)
end