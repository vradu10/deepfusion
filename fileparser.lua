--[[
Data File Parser class to read from file
By Valentin Radu @ University of Edinburgh
--]]

local FileParser = torch.class('FileParser')

local PAGING_SIZE = 10000	-- used to limit the number of lines read from file to table (LuaJIT memory limitation)

function FileParser:__init(config)
	self.name = 'FileParser'

	self.config = config
end

function FileParser:parseDataFile()
	if self.config.validation_mode == "leave-one-out" or 
			self.config.validation_mode == "train-test" then
		self:fetchSeparateTrainTest(
			self.config.trainFile, 
			self.config.testFile)
	else
		-- if cross-validation
		self:fetchDataCrossValidation(
			self.config.file, 
			self.config.folds, 
			self.config.current_fold)
	end
	collectgarbage()
	return self.data
end

-- preparing data for leave-one-out evaluation method
function FileParser:fetchSeparateTrainTest(trainFile, testFile)
	if (trainFile == nil or testFile == nil) then
		print ("Please provide the train file name and the test file name")
		return nil
	end

	self.data = {}
	self.data['trainData'], self.data['trainLabel'], mean, std = 
			self:readFile(self.config.path .. trainFile, nil, nil)
	self.data['testData'], self.data['testLabel'], _, _ = 
			self:readFile(self.config.path .. testFile, mean, std)
	
	self.data['classes'] = self.config.classes
	return self.data
end

-- preparing data for cross-validation evaluation method
function FileParser:fetchDataCrossValidation(fileName, noFolds, indexFoldValidation)
	local noFolds = noFolds or 10
	local indexFoldValidation = indexFoldValidation or 1

	if (fileName == nil) then
		print ("File name for cross-validation parsing is missing -- please privide the name of this file.")
		return nil
	end

	print("reading file: " .. self.config.path .. fileName)
	featuresTensor, labelsTensor, _, _ = self:readFile(self.config.path .. fileName, nil, nil)

	print('splitting folds..')

	-- make view for test subset
	noLines = featuresTensor:size(1)

	self.data = {}			-- data structure that will hold the file information

	foldSize = math.floor(noLines / noFolds)

	foldsDataTable = {}
	foldsLabelTable = {}
	for i = 1, noFolds - 1 do
		foldsDataTable[#foldsDataTable + 1] = featuresTensor:narrow(1, (i-1) * noFolds + 1, foldSize):clone()
		foldsLabelTable[#foldsLabelTable + 1] = labelsTensor:narrow(1, (i-1) * noFolds + 1, foldSize):clone()
	end

	lastFoldSize = noLines - (noFolds - 1) * foldSize 

	foldsDataTable[#foldsDataTable + 1] = featuresTensor:narrow(1, noLines - lastFoldSize + 1, lastFoldSize):clone()
	foldsLabelTable[#foldsLabelTable + 1] = labelsTensor:narrow(1, noLines - lastFoldSize + 1, lastFoldSize):clone()

	-- use indexFoldValidation as the fold for 
	self.data['testData'] = foldsDataTable[indexFoldValidation]
	self.data['testLabel'] = foldsLabelTable[indexFoldValidation]
	
	-- select a starting point for iterating over training data
	local trainingStartIndex = 1
	if indexFoldValidation == 1 then
		trainingStartIndex = 2
	end
	self.data['trainData'] = foldsDataTable[trainingStartIndex]
	self.data['trainLabel'] = foldsLabelTable[trainingStartIndex]

	-- concatenate the folds for training data
	for i = 1, noFolds do
		if (i ~= indexFoldValidation and i ~= trainingStartIndex) then
			self.data['trainData'] = torch.cat(self.data['trainData'], foldsDataTable[i], 1)
			self.data['trainLabel'] = torch.cat(self.data['trainLabel'], foldsLabelTable[i], 1)
		end
	end

	self.data['classes'] = self.config.classes
	
	return self.data
end


function FileParser:readFile(fileName, mean, std)
	
	local signals = {}	-- [noOfSignals][indexInWindow]
	local no_signals = #self.config.input_splits
	for i = 1, no_signals do
		signals[i] = {}
	end

	local labels = {}

	-- open file for read
	fileIn,err = io.open(fileName)
	if err then 
		print(fileName.. " could not be opened. Please check file exists.")
		return
	end

	signalsTensorSet = {}	-- set of tensors holding time-window features, converted from signals table

	lineNo = 0
	-- line by line
	while true do

		line = fileIn:read()
		lineNo = lineNo + 1

		if (line == nil or lineNo % PAGING_SIZE == 0) then
			if (signalsTensorSet[1] == nil) then
				-- if this is the first page of the tensor, initialise the tensor
				for i = 1, no_signals do
					signalsTensorSet[i] = torch.Tensor(signals[i])
				end
				labelsTensor = torch.Tensor(labels)
			else
				-- concatenate to previous pages
				for i = 1, no_signals do
					signalsTensorSet[i] = torch.cat(signalsTensorSet[i], torch.Tensor(signals[i]), 1)
				end
				labelsTensor = torch.cat(labelsTensor, torch.Tensor(labels), 1)
			end

			-- clear previous page from running table signals
			for i = 1, no_signals do
				signals[i] = {}
			end
			labels = {}
			collectgarbage()
			print("paging...")
		end

		-- 'while' termination condition
		if (line == nil) then
			break
		end

		-- split line into elements
		atoms = splitLine(line)

		signalsLine = {}
		for i = 1, no_signals do
			signalsLine[i] = {}
		end

		local atomsIndex = 1
		for i = 1, no_signals do
			for j = 1, self.config.input_splits[i] do
				signalsLine[i][j] = atoms[atomsIndex]
				atomsIndex = atomsIndex + 1
			end
		end
		
		-- add this sensor line to the table
		for i = 1, no_signals do
			signals[i][#signals[i] + 1] = signalsLine[i]
		end
		labels[#labels + 1] = atoms[#atoms]
	end

	print('went throuhg the file')

	-- check if we need other features type
	if (self.config.preprocess == "freq") then 
		print("In Freq domain")
		signal = require 'signal'

		for i = 1, no_signals do
			for j = 1, signalsTensorSet[i]:size(1) do
				signalsTensorSet[i][j]:copy(signal.fft(signalsTensorSet[i][j]):narrow(2,1,1))
			end
		end
	else
		-- as they are
		print("In time domain")
	end


	if (self.config.preprocess == "ecdf") then
		print("to ECDF features...")

		for i = 1, no_signals do
			signalsTensorSetECDF = torch.Tensor(signalsTensorSet[i]:size(1), self.config.ecdf_components)
			for j = 1, signalsTensorSet[i]:size(1) do
				signalsTensorSetECDF[j]:copy(self:ecdfRep(signalsTensorSet[i][j]))
			end
			signalsTensorSet[i] = signalsTensorSetECDF
		end
		collectgarbage()
	end

	-- normalise
	if (mean == nil) then
		-- if there is no mean and std provided, compute them
		mean = {}
		std = {}
		for i = 1, no_signals do
			mean[i] = signalsTensorSet[i]:mean()
			std[i]  = signalsTensorSet[i]:std()
		end
	end

	for i = 1, no_signals do
		signalsTensorSet[i]:add(-mean[i]):div(std[i])
	end
	
	-- concatenate
	local featuresTensor = signalsTensorSet[1]
	for i = 2, no_signals do
		featuresTensor = torch.cat(featuresTensor, signalsTensorSet[i], 2)
	end

	if (self.config.class_indexed_zero) then
		labelsTensor:add(1)	-- the index of classes should start with 1, not 0
	end

	featuresTensor = featuresTensor:contiguous()
	collectgarbage()

	return featuresTensor, labelsTensor, mean, std
end

function FileParser:generateARFF(which)
	local which = which or 1 	-- 1 training data, 2 - evaluation data

	local dataset = {}
	dataset['data'] = self.data['trainData']	-- the training data
	dataset['label'] = self.data['trainLabel']
	
	if which == 2 then
		-- for test data
		dataset['data'] = self.data['testData']
		dataset['label'] = self.data['testLabel']
	end
	dataset['classes'] = self.data['classes']


	print("preparing ARFF file ...")
	f = io.open("classifiers_" .. which .. ".arff", 'w')
	f:write("@relation rel\n\n")


	for i = 1, dataset['data']:size(2) do
		f:write("@attribute x"..i.." numeric\n")
	end

	f:write("@attribute class {")
	for i = 1, #dataset['classes'] - 1 do
		f:write(i .. ".0,")
	end
	f:write(#dataset['classes'] .. ".0}\n")
	f:write("\n@data\n")


	for i = 1, dataset['data']:size(1) do
		for j = 1, dataset['data']:size(2) do
			f:write(dataset['data'][i][j] .. ",")
		end
		f:write(dataset['label'][i] .. "\n")
	end
	f:close()
	print("wrote ARFF file.")
	collectgarbage()
end

-- tokenize a string on comma separator
function splitLine(line,sep)
	local sep = sep or ","
	local data = {}
	for word in string.gmatch(line, "([^".. sep .. "]+)") do 	-- like ([^,]+)
		table.insert(data, word)
	end
	return data
end

-- according to https://github.com/nhammerla/ecdfRepresentation/blob/master/torch/ecdfRep.lua
function FileParser:ecdfRep(data)
	local d = data:clone()
	if (d:nDimension() > 1) then
		if (d:size(2) > d:size(1)) then
			d = d:t()
		end
	else
		d:resize(d:size(1),1)
	end
	
	-- get indeces for sorted array (ascending)
	d = d:sort(1)
	
	-- index the right elements and add mean (transpose for compatibility with matlab etc)
	e = d:index(1, 
		torch.linspace(1, d:size(1), self.config.ecdf_components):round():long()):t():reshape(1,self.config.ecdf_components)
	
	return e
end
