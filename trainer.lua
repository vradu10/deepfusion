--[[
Trainer class to train the model
By Valentin Radu @ University of Edinburgh
--]]

require 'nn'
require 'optim'
require 'optim_updates'

require 'tester'

local Trainer = torch.class('Trainer')

function Trainer:__init(config)	
	self.name = 'Trainer'
	if not config then
		os.error("Trainer no configuration provided")
	end

	self.config = config

	-- check if running on GPU
	self.onGPU = false
	self.config.train.criterion:type(self.config.main.type)
	self.config.model:type(self.config.main.type)

	if self.config.main.gpuid > -1 then	-- if running on GPU
		self.onGPU = true
		require 'cunn'	
	end
end

function Trainer:run(dataLoader)
	-- reduce namespace lenght
	local train = self.config.train

	train.last_errors = train.last_errors or {}	-- used to compute the sum of last n errors
	train.bestTestAccuracy = 0
	train.bestF1Score = 0
	train.current_epoch = train.current_epoch or 0

	local tester = Tester(self.config)

	train.optim_state = {}

	-- loop over epochs
	while true do
		train.current_epoch = train.current_epoch + 1

		if self:earlyStop(train.last_errors) or 
			train.current_epoch > train.epochs then
			break
		end

		print('\n### EPOCH = ' .. train.current_epoch .. ' ###\n')
		
		-- train over dataset
		self:train(dataLoader)

		-- test performance of current model
		testAccuracy, f1Score, confusion = tester:test(self.config.model, dataLoader, self.onGPU)

		-- determine best F1 Score
		if f1Score > train.bestF1Score then 
			train.bestF1Score = f1Score
			self:saveStatus(true, confusion)
		end

		-- determine best accuracy
		if testAccuracy > train.bestTestAccuracy then
			train.bestTestAccuracy = testAccuracy
		end

		-- store current test accuracy in the last error array
		train.last_errors[train.current_epoch % train.no_last_errors + 1] = testAccuracy

		collectgarbage()

		-- save current progress once in a while
		if train.current_epoch % self.config.main.save_freq_epoch == 0 then
			self:saveStatus()
		end

	end

	return train.bestTestAccuracy, train.bestF1Score
end

function Trainer:train(dataLoader)
	-- reduce namespace
	train = self.config.train

	-- retrieving model parameters and gradients
	local parameters, gradParameters = self.config.model:getParameters()

	-- confusion matrix 
	local confusion = optim.ConfusionMatrix(dataLoader.data['classes'])
	local optim_state = {}

	local batchData = nil
	local batchLabels = nil
	
	-- if self.onGPU then cutorch.synchronize() end
	local time = sys.clock()

	self.config.model:training()

	dataLoader:resetTrainBatches()

	-- Looping over all training data
	for i = 1, dataLoader:getNoTrainBatches() do

		-- load batch data
		batchData, batchLabels = dataLoader:getTrainBatch(batchData, batchLabels)

		-- move to GPU if so
		if self.onGPU then
			batchData:cuda()
			batchLabels:cuda()
		end


		-- evaluareating f(X) and df/dX
		local feval = function()

--			if train.fine_tune_after >= 0 and train.current_epoch >= train.fine_tune_after then
			gradParameters:zero()

			-- Forward pass
			local output = self.config.model:forward(batchData)
			local loss = train.criterion:forward(output, batchLabels)	-- tensor may need to be created

			-- Backwards pass
			local dloss_doutput = train.criterion:backward(output, batchLabels)	-- gradOutput
			local dx = self.config.model:backward(batchData, dloss_doutput)
		
			confusion:batchAdd(output, batchLabels)

			-- apply L2 regularization
			gradParameters:add(train.opt.weightDecay, parameters)

			-- clamp odd values
			gradParameters:clamp(-train.grad_clip, train.grad_clip)

			return loss, gradParameters

		end


		-- -- decay the learning rate
		local learning_rate = train.opt.learningRate
		if i > train.opt.learning_rate_decay_start and train.opt.learning_rate_decay_start >= 0 then
			local frac = (train.current_epoch - train.opt.learning_rate_decay_start) / train.opt.learning_rate_decay_every
			local decay_factor = math.pow(0.5, frac)
			learning_rate = learning_rate * decay_factor -- set the decayed rate
		end

		local f = feval()

		-- perform a parameter update
		if train.optimization_method == 'sgd' then
			sgd(parameters, gradParameters, learning_rate)
		elseif train.optimization_method == 'sgdm' then
			sgdm(parameters, gradParameters, learning_rate, train.opt.optim_alpha, train.optim_state)
		elseif train.optimization_method == 'adam' then
			adam(parameters, gradParameters, learning_rate, train.opt.optim_alpha, train.opt.optim_beta, train.opt.optim_epsilon, train.optim_state)
		else
			error('bad option opt.optim')
		end

		-- print progress
		xlua.progress(i * batchData:size(1), dataLoader:getNoTrainBatches() * batchData:size(1))

	end




	-- if self.onGPU then cutorch.synchronize() end		-- if care about the actual time uncomment line but slower
	time = sys.clock() - time
	time = time / dataLoader:getNoTrainInstances()

	print("[*] Time to learn 1 sample = " .. (time * 1000) .. 'ms')

	-- Confusion matrix
	print(confusion) print('\n')
	confusion:zero()

	print('[*] Best test model acc (so far) = ' .. train.bestTestAccuracy)
	print('[*] Best test model F1-score (so far) = ' .. train.bestF1Score)

	self.config.model:evaluate()
end

function Trainer:saveStatus(best, confusion)
	local fileName = self.config.main.saves_folder .. 
						"config_" .. os.date('%d-%b-%Y-%H-%M-%S') .. ".save"

	if best then
		-- save confusion matrix
		torch.save(self.config.main.saves_folder .. "fold_" .. self.config.data.current_fold .. "_best_matrix_" .. config.main.runtime .. ".save", confusion)
		-- give a special name for the current configuration
		fileName = self.config.main.saves_folder .. "fold_" .. self.config.data.current_fold .. "_best_config_" .. config.main.runtime .. ".save"
	end

	torch.save(fileName, self.config)
end

-- determines if there is an early stop "true" or the sum of previous n errors is greater than stop "false"
function Trainer:earlyStop()
	-- check early stop if error rate over the last n elementes has been low
	if #self.config.train.last_errors < config.train.no_last_errors then
		return false
	end
	local sum_err = 0 
	for _,err in pairs(self.config.train.last_errors) do
		sum_err = sum_err + err
	end
	if sum_err < config.train.early_stop_error then
		return true
	end
	return false
end
