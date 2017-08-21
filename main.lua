--[[
Main script to run the training -- check configurations in config.lua
By Valentin Radu @ University of Edinburgh
--]]

package.path = package.path .. ";../?.lua"

--require 'datareader'
require 'dataloader'
require 'trainer'
require 'model'


-- Configurations file -- all settings are in this file
dofile("config_activity.lua")
--dofile("config_gait.lua")

local main = {}	-- namespace

function main.main()

	-- parse command line paramteres
	local configurationString = main.parse_args()
	print(configurationString)

	-- create results folder if needed
	main.make_folder(config.main.results_folder)	-- holds the evaluation results
	main.make_folder(config.main.saves_folder)		-- holds checkpoint saves
	
	-- current system time for identifying current run
	config.main.runtime = os.date('%d-%b-%Y-%H-%M-%S')

	-- construct log file name
	main.logFileName = config.main.results_folder .. 
			"log_" .. config.main.name ..
			"-" .. config.main.arch_index .. 
			"_" .. config.data.batch_size .. 
			"_" .. config.data.preprocess .. "_" .. 
			config.main.runtime ..".txt"

	logFile = io.open(main.logFileName, 'a')

	-- write configuratiom text to log file
	logFile:write(configurationString .. "\n")
	logFile:close()

	doWork()

end

function doWork()
	config.main.sumBestTest = 0
	config.main.sumBestF1Score = 0

	-- if evaluation mode is "leave-one-out", treat individuals as folds in the main loop below
	if config.data.validation_mode == "leave-one-out" then
		config.data.starting_fold = config.data.starting_element
		config.data.folds = #config.data.elements
	end

	-- create model from scratch
	config.init_model()
	local model = Model(config.model)
	
	model:initializeParams2(config.main.params_randomize)

	-- with pretrained model
	if config.main.model_file then
		config.model = torch.load(config.main.model_file)
	end

	-- check if resuming an older task
	if config.main.resume then
		config = torch.load(config.main.resume)
	end

	print(config.model)

	-- determine first element in the loop, important when resuming an older task
	local starting_fold = config.data.starting_fold
	if config.main.current_fold then
		starting_fold = config.main.current_fold
	end

	local ending_fold = config.data.ending_fold or (config.data.folds + config.data.starting_fold - 1)

	if config.data.validation_mode == "train-test" then
		starting_fold = 1
		ending_fold = 1
	end

	-- the main loop over the folds -- repeat the initiation and traing for each fold
	for task_index = starting_fold, ending_fold do

		print("Training fold " .. task_index .. " ...")

		-- save current fold - useful for resuming training
		config.main.current_fold = task_index

		-- set the train and test file names in the case of leave one out evaluation mode
		if config.data.validation_mode == "leave-one-out" then
			config.data.trainFile = "without_" .. config.data.elements[task_index] .. ".arffx"
			config.data.testFile = "with_" .. config.data.elements[task_index] .. ".arffx"
		end

		-- data fold different than loop index due to wrapping of loop index (to allow starting_fold is not 1)
		local dataFoldIndex = (task_index - 1) % config.data.folds + 1
		config.data.current_fold = dataFoldIndex

		local dataLoader = DataLoader(config.data)
	
		config.train.current_epoch = 0	
		local trainer = Trainer(config)
		local bestAccuracyTest, bestF1Score = trainer:run(dataLoader)

		config.main.sumBestTest = config.main.sumBestTest + bestAccuracyTest
		config.main.sumBestF1Score = config.main.sumBestF1Score + bestF1Score
		
		main.logPerformance(config.main.current_fold, bestAccuracyTest, bestF1Score)

		-- prepare the model for next iteration, replace all parameters with random values before next training
		model:initializeParams(config.main.params_randomize)
	end

	main.logPerformance("AVERAGE", (config.main.sumBestTest / (ending_fold - starting_fold + 1)), 
			(config.main.sumBestF1Score / (ending_fold - starting_fold + 1)))

end

function main.logPerformance(iteration, accuracy, score)
	print("Status update @ " .. iteration .. " - Accuracy: " .. accuracy .. " F1-score: " .. score)
	log = io.open(main.logFileName ,'a')
	log:write("after iteration -> " .. iteration .. "\n")
	log:write("average Accuracy, " .. accuracy .. '\n')
	log:write("average F1Score, " .. score .. '\n')
	log:close()
end

-- parse command line parameters and update config variables
function main.parse_args()
	-- read input parameters
	cmd = torch.CmdLine()
	cmd:text("Setting the network configuration")
	cmd:text()
	cmd:text("Options")
	cmd:option("-batch_size", config.data.batch_size, "batch size")
	cmd:option("-net", config.main.arch_index, "network architecture index")	
	cmd:option("-evaluation", config.data.validation_mode, "type of evaluation: cross-validation | leave-one-out")
	cmd:option("-dataPreprocess", config.data.preprocess, "Data preprocessing: time | freq | ecdf")
	cmd:option("-startingFold", config.data.starting_fold, "folds to start with")
	cmd:option("-endingFold", config.data.ending_fold, "folds to end with")
	cmd:option("-lr", config.train.opt.learningRate, "learning rate")
	cmd:option("-momentum", config.train.opt.momentum, "momentum")
	cmd:option("-dropout", config.train.dropout_p, "dropout probability")
	cmd:option("-epochs", config.train.epochs, "no of supervised learning epochs")
	cmd:option("-resume", "", "resume training from file")
	cmd:option("-model", "", "continue training initial model")
	cmd:option("-gpuid", config.main.gpuid, "which GPU to use; -1 is CPU")
	cmd:option("-seed", config.main.seed, "set seed for random number generation")
	cmd:option("-dataset_fraction", config.data.dataset_fraction, "fraction of data used in training and test from original size in files")
	cmd:option("-verbose", config.main.verbose, "verbose level: 0 - evaluation; 1 - debug")
	cmd:option("-optim", config.train.optimization_method, "optimization method: sgd | adam | sgdm ")
	cmd:text()

	-- parse input params
	params = cmd:parse(arg)


	-- warning: no checks for input parameters
	config.data.batch_size = params.batch_size
	config.main.arch_index = params.net
	config.data.validation_mode = params.evaluation
	config.data.preprocess = params.dataPreprocess
	config.data.starting_fold = params.startingFold
	config.data.ending_fold = params.endingFold
	config.train.opt.learningRate = params.lr
	config.train.opt.momentum = params.momentum
	config.train.dropout_p = params.dropout
	config.train.epochs = params.epochs
	config.main.gpuid = params.gpuid
	config.main.seed = params.seed
	config.data.dataset_fraction = params.dataset_fraction
	config.main.verbose = params.verbose
	config.train.optimization_method = params.optim

	-- determine if this should run on the CPU or GPU

	-- assuming first that this will be running o the CPU
	config.main.type = "torch.DoubleTensor"
	torch.manualSeed(config.main.seed)
	
	if config.main.gpuid > -1 then
		-- checking if this should be running on the GPU with CUDA

		local ok_cunn, cunn = pcall(require, 'cunn')
    	local ok_cutorch, cutorch = pcall(require, 'cutorch')
		
		if ok_cunn and ok_cutorch then
			cutorch.setDevice(config.main.gpuid + 1) -- +1 to make it 0 indexed
			cutorch.manualSeed(config.main.seed)
		else
			print ("cunn or cutorch not installed. Continuing on CPU")
			config.main.type = "torch.DoubleTensor"
		end
	end

	-- determine if should resume
	if params.resume ~= "" then
		print("Resuming training from file " .. params.resume)
		config.main.resume = params.resume	-- the file holding the preliminary configuration
	end

	if params.model ~= "" then
		print("Using pre-trained model from " .. patams.model)
		config.main.model_file = params.model
	end

	-- prepare a string with the instructed configuration of this task
	local str = "Batch size: " .. config.data.batch_size .. "\n" ..
		"Learning Rate: " .. config.train.opt.learningRate .. "\n" ..
		"Momentum: " .. config.train.opt.momentum .. "\n" ..
		"Dropout: " .. config.train.dropout_p .. "\n" ..
		"Network Arch Type: " .. config.main.arch_index .. "\n" ..
		"No of epochs: " .. config.train.epochs .. "\n" ..
		"Evaluation mode: " .. config.data.validation_mode .. "\n" ..
		"Optimization: " .. config.train.optimization_method .. "\n" ..
		"Data pre-processing: " .. config.data.preprocess
	if config.data.starting_fold ~= 1 then
		srt = str .. "\nStarting with fold: " .. config.data.starting_fold
		srt = str .. "\nEnding with fold: " .. config.data.ending_fold
	end

	return str
end

function main.make_folder(folderName)
	-- Quering for a result folder
	folderName = folderName or 'results'
	folderPresent = paths.dir(folderName)

	if not folderPresent then
		-- The result folder does not exist and creating a new result folder
		status = paths.mkdir(folderName)
		if status then
			print("Folder " .. folderName .. "succesfully created.")
		else
			error('Folder can not be created.')
		end
	end
end

main.main()
