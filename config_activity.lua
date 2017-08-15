--[[
Configuration for Activity Training Program
By Valentin Radu @ University of Edinburgh
--]]

package.path = package.path .. ";../?.lua"

require 'nn'
require 'parallelseq'

-- The namespace
config = {}


-- Main program
config.main = {}
config.main.CPU_type = "torch.DoubleTensor"		-- GPU is CudaTensor by default
config.main.gpuid = -1 			-- which GPU to use [0, 1, ...] and -1 is for CPU
config.main.arch_index = 1 		-- ** model achitecture - see below for the exact structure
config.main.autoencoder = false
config.main.unsupervised_epochs = 10
config.main.seed = 123
config.main.params_randomize = 8e-2 -- 5e-2
config.main.dropout = true
config.main.save_model = true		-- saves the current model every save_freq_epoch to facilitate resume training
config.main.save_freq_epoch = 50 	-- how often to save progress on number of epochs
config.main.device = 1
config.main.test = true
config.main.name = "act_rec"
config.main.verbose = 0
config.main.show_graphics = false

config.main.results_folder = "results/"
config.main.saves_folder = "saved/"


-- Data
config.data = {}
config.data.batch_size = 4
config.data.preprocess = "time"									-- "time" or "freq" or "ecdf"
config.data.validation_mode = "leave-one-out"					-- "cross-validation" or "leave-one-out"
config.data.input_splits = {300, 300}
config.data.classes = {"stand", "walk", "stairsdown", "bike", "stairsup", "sit"}
config.data.elements = {'a','b','c','d','e','f','g','h','i'}	-- for leave one out evaluation mode
config.data.starting_element = 1					-- first element used for testing, training all the others except this one
config.data.folds = 10
config.data.starting_fold = 1
config.data.ending_fold = config.data.folds
config.data.path = "../../../leave_one_user_out_3D_acc_3D_gyro/"
config.data.file = "all.arff0"
config.data.class_indexed_zero = true
config.data.ecdf_components = 30
config.data.dataset_fraction = 1--0.5 					-- percentage of file data used in the experiment 

-- Training
config.train = {}
config.train.dropout_p = 0.4
config.train.fine_tune_after = 0 					-- 0 fine tune from the start; -1 never fine tune
config.train.grad_clip = 0.1
config.train.optimization_method = 'sgd'	-- this takes 'sgd', 'adam', ''
config.train.epochs = 5000		-- total number of epochs used for training
config.train.criterion = nn.ClassNLLCriterion()		-- loss function criterion
config.train.early_stop_error = 1e-6
config.train.no_last_errors = 5

config.train.opt ={}
config.train.opt.learningRate = 3e-3 --1e-5		-- 300 times less for 'adam'
config.train.opt.learningRateDecay = 5e-7
config.train.opt.weightDecay = 0
config.train.opt.momentum = 1e-2
config.train.opt.beta1 = 0.8 	-- first adam parameter
config.train.opt.beta2 = 0.999	-- second adam parameter
config.train.opt.learning_rate_decay_start = -1 	-- at what iteration to start decaying learning rate? (-1 = dont)
config.train.opt.learning_rate_decay_every = 50000 	-- every how many iterations thereafter to drop LR by half?
config.train.opt.optim_epsilon = 1e-8				-- epsilon that goes into denominator for smoothing


-- determine the number of features in input data to construct net structure
config.data.input_size = 0
for i = 1, #config.data.input_splits do
	config.data.input_size = config.data.input_size + config.data.input_splits[i]
end

-- The model based on the main.arch_index variable set above


function config.init_model()
	if config.model then
		return config.model
	end

	config.model = nn.Sequential()

	if config.main.arch_index == 1 then
		-- parallel architecture
		local parallel = ParallelSeq(config.data.input_splits)

		for i = 1, #config.data.input_splits do
			-- populate ParallelSeq for each modality branch

			parallel:add(i, nn.Linear(config.data.input_splits[i], config.data.input_splits[i]))
			parallel:add(i, nn.BatchNormalization(config.data.input_splits[i]))
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.Dropout(config.train.dropout_p))
			parallel:add(i, nn.Linear(config.data.input_splits[i], config.data.input_splits[i]))
			parallel:add(i, nn.BatchNormalization(config.data.input_splits[i]))
		end

		config.model:add(parallel)
		config.model:add(nn.Linear(config.data.input_size, config.data.input_size))
		config.model:add(nn.BatchNormalization(config.data.input_size))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size, #config.data.classes))
		--config.model:add(nn.BatchNormalization(#config.data.classes))
		config.model:add(nn.LogSoftMax())

	elseif config.main.arch_index == 2 then
		-- the smaller parallel architecture
		local parallel = ParallelSeq(config.data.input_splits)

		for i = 1, #config.data.input_splits do
			-- populate ParallelSeq for each modality branch

			parallel:add(i, nn.Linear(config.data.input_splits[i], config.data.input_splits[i] / 2))
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.Dropout(config.train.dropout_p))
			parallel:add(i, nn.Linear(config.data.input_splits[i] / 2, config.data.input_splits[i] / 2))
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.Dropout(config.train.dropout_p))
			parallel:add(i, nn.Linear(config.data.input_splits[i] / 2, config.data.input_splits[i] / 2))
		end

		config.model:add(parallel)
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size / 2, config.data.input_size / 2))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size / 2, #config.data.classes))
		config.model:add(nn.LogSoftMax())

	elseif config.main.arch_index == 3 then
		-- the larger parallel architecture
		local parallel = ParallelSeq(config.data.input_splits)

		for i = 1, #config.data.input_splits do
			-- populate ParallelSeq for each modality branch

			parallel:add(i, nn.Linear(config.data.input_splits[i], config.data.input_splits[i] * 2))
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.Dropout(config.train.dropout_p))
			parallel:add(i, nn.Linear(config.data.input_splits[i] * 2, config.data.input_splits[i] * 2))
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.Dropout(config.train.dropout_p))
			parallel:add(i, nn.Linear(config.data.input_splits[i] * 2, config.data.input_splits[i] * 2))
		end

		config.model:add(parallel)
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size * 2, config.data.input_size * 2))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size * 2, #config.data.classes))
		config.model:add(nn.LogSoftMax())

	elseif config.main.arch_index == 5 then
		-- the simple (non-branching) DNN arch
		config.model:add(nn.Linear(config.data.input_size, config.data.input_size))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size, config.data.input_size))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size, config.data.input_size))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size, config.data.input_size))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(config.data.input_size, #config.data.classes))
		config.model:add(nn.LogSoftMax())

	elseif config.main.arch_index == 6 then
		-- the bracnched CNN
		local parallel = ParallelSeq(config.data.input_splits)

		for i = 1, #config.data.input_splits do
			-- populate ParallelSeq for each modality branch

			parallel:add(i, nn.Reshape(config.data.input_splits[i],1))
			parallel:add(i, nn.TemporalConvolution(1,64,7))	-- in: 1x300, out: 128x294
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.TemporalMaxPooling(3,3))			-- in: 128x294, out: 128x98
			parallel:add(i, nn.TemporalConvolution(64,32,3)) 	-- in: 128x98,  out: 128x96
			parallel:add(i, nn.ReLU())
			parallel:add(i, nn.TemporalMaxPooling(3,3))			-- in: 128x96,  out: 128x32
		end

		config.model:add(parallel)
		
		config.model:add(nn.Reshape(32 * 32 * #config.data.input_splits))
		config.model:add(nn.Linear(32 * 32 * #config.data.input_splits, 128))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(128,#config.data.classes))
		config.model:add(nn.LogSoftMax())

	elseif config.main.arch_index == 7 then
		-- the simple CNN
		config.model:add(nn.Reshape(config.data.input_size,1))
		config.model:add(nn.TemporalConvolution(1,64,7))		-- in: 1x600, out: 128x594
		config.model:add(nn.ReLU())
		config.model:add(nn.TemporalMaxPooling(3,3))			-- in: 128x594, out: 128x198
		config.model:add(nn.TemporalConvolution(64,32,3)) 	-- in: 128x198, out: 128x196
		config.model:add(nn.ReLU())
		config.model:add(nn.TemporalMaxPooling(3,3))			-- in: 128x196, out: 128x65
		config.model:add(nn.Reshape(32 * 65))
		config.model:add(nn.Linear(32 * 65, 128))
		config.model:add(nn.ReLU())
		config.model:add(nn.Dropout(config.train.dropout_p))
		config.model:add(nn.Linear(128,#config.data.classes))
		config.model:add(nn.LogSoftMax())
	end
	return config.model
end
