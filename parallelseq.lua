--[[
Parallel Sequence with branches for each modality
By Valentin Radu @ University of Edinburgh
--]]

require 'nn'

local ParallelSeq, parent = torch.class('ParallelSeq', 'nn.Container')

-- This class create a container with parallel branches.
-- The difference between this and the Parallel class is that inputs 

function ParallelSeq:__init(inputSplits, inputDimension, outputDimension)
   parent.__init(self)

   self.modules = {}
   self.inputDimension = inputDimension or 1
   self.outputDimension = outputDimension or 1
   self.inputSplits = inputSplits or {1,1,1} -- input size of each branch
   
   self.inputOffsets = {}  -- offset in the linear input for each branch
   local offset = 1
   for i = 1, #self.inputSplits do
      self.inputOffsets[i] = offset
      offset = offset + self.inputSplits[i]
   end
end


function ParallelSeq:add(component)
   print ('Branch index needs to be specified. Use * add(branch_index, component) *')
end


function ParallelSeq:add(branchIndex, component)

   if (not component) then
      print ('Use * add(branch_index, component) *')
      return
   end

   if (branchIndex < 1 or branchIndex > #self.inputSplits) then
      print ('Please insert modules in order of their branch index')
      return
   end

   -- initialise a sequential on the new branch
   if (not self.modules[branchIndex]) then 
      self.modules[branchIndex] = nn.Sequential()
   end

   -- add the new component to the proper sequence
   local branchSequence = self.modules[branchIndex]
   branchSequence:add(component)
end


function ParallelSeq:getNoBranches()
   return #self.inputSplits
end


function ParallelSeq:getBranchInput(input, branchIndex)
   -- select the area of input specific to this branch
   
   return input:narrow(
      self.inputDimension, 
      self.inputOffsets[branchIndex], 
      self.inputSplits[branchIndex]
   )
end


-- *******************************
-- training util methods below: 

function ParallelSeq:training()
   for i = 1, self:getNoBranches() do
      self.modules[i]:training()
   end
end

function ParallelSeq:evaluate()
   for i = 1, self:getNoBranches() do
      self.modules[i]:evaluate()
   end
end

function ParallelSeq:updateOutput(input)

   if input:dim() > 1 then
      self.inputDimension = 2
      self.outputDimension = 2
   end

   local outputsArray = {}   -- array of outputs for each branch

   -- total length of the output on the self.outputDimension
   self.totalOutputSize = self.totalOutputSize or torch.LongStorage()
   local totalOutputSize = self.totalOutputSize

   -- for each parallel branch do a sequential pass
   for i = 1, self:getNoBranches() do
      local branchInput = self:getBranchInput(input, i)
      local branchOutput = self.modules[i]:updateOutput(branchInput)
      local branchOutputSize = branchOutput:size(self.outputDimension)  

      outputsArray[i] = branchOutput

      -- increase the totalOutputSize with the size of current output
      if i == 1 then
         totalOutputSize:resize(branchOutput:dim()):copy(branchOutput:size())
      else
         totalOutputSize[self.outputDimension] = totalOutputSize[self.outputDimension] + branchOutputSize
      end
   end

   -- copy the output of each branch into the final output
   self.output:resize(totalOutputSize)

   local outputOffset = 1  -- offset of each branch output in the final output

   for i = 1, self:getNoBranches() do
      local branchOutput = outputsArray[i]
      local branchOutputSize = branchOutput:size(self.outputDimension)
      self.output:narrow(
               self.outputDimension, 
               outputOffset, 
               branchOutputSize
         ):copy(branchOutput)
      outputOffset = outputOffset + branchOutputSize
   end

   return self.output
end


function ParallelSeq:updateGradInput(input, gradOutput)
   local outputOffset = 1
   local gradInputsArray = {}

   -- iterate over each branch
   for i = 1, self:getNoBranches() do

      -- select the area of input specific to this branch
      local branchInput = self:getBranchInput(input, i)
      local branch = self.modules[i]
      local branchOutput = branch.output
      local branchOutputSize = branchOutput:size(self.outputDimension)
      local branchGradOutput = gradOutput:narrow(
                                    self.outputDimension, 
                                    outputOffset, 
                                    branchOutputSize
                              )
      outputOffset = outputOffset + branchOutputSize

      local branchGradInput = branch:updateGradInput(branchInput, branchGradOutput)
      
      gradInputsArray[i] = branchGradInput
   end

   -- make sure self.gradInput is the proper size
   self.gradInput:resizeAs(input)
   
   -- copy the values of gradInput for each branch into the global gradInput
   for i = 1, self:getNoBranches() do

      self.gradInput:narrow(self.inputDimension, 
                              self.inputOffsets[i], 
                              self.inputSplits[i]
                        ):copy(gradInputsArray[i])
   end

   return self.gradInput
end


function ParallelSeq:accGradParameters(input, gradOutput, scale)
   local outputOffset = 1

   for i = 1, self:getNoBranches() do
      -- select the area of input specific to this branch
      local branchInput = self:getBranchInput(input, i)
      local branch = self.modules[i]
      local branchOutput = branch.output
      local branchOutputSize = branchOutput:size(self.outputDimension)

      branch:accGradParameters(
         branchInput,
         gradOutput:narrow(self.outputDimension, outputOffset, branchOutputSize),
         scale
      )
      outputOffset = outputOffset + branchOutputSize
   end
end


function ParallelSeq:accUpdateGradParameters(input, gradOutput, lr)
   local outputOffset = 1

   for i = 1, self:getNoBranches() do
      -- select the area of input specific to this branch
      local branchInput = self:getBranchInput(input, i)
      local branch = self.modules[i]
      local branchOutput = branch.output
      local branchOutputSize = branchOutput:size(self.outputDimension)
      
      branch:accUpdateGradParameters(
          branchInput,
          gradOutput:narrow(self.outputDimension, outputOffset, branchOutputSize),
          lr
      )  
      outputOffset = outputOffset + branchOutputSize
   end
end

function ParallelSeq:backward(input, gradOutput, scale)
   
   local outputOffset = 1
   local gradInputsArray = {}

   -- iterate over each branch
   for i = 1, self:getNoBranches() do

      -- select the area of input specific to this branch
      local branchInput = self:getBranchInput(input, i)
      local branch = self.modules[i]
      local branchOutput = branch.output
      local branchOutputSize = branchOutput:size(self.outputDimension)
      local branchGradOutput = gradOutput:narrow(
                                    self.outputDimension, 
                                    outputOffset, 
                                    branchOutputSize
                              )
      outputOffset = outputOffset + branchOutputSize  

      local branchGradInput = branch:backward(
                                    branchInput, 
                                    branchGradOutput, 
                                    scale
                              )
      gradInputsArray[i] = branchGradInput
   end

   -- make sure self.gradInput is the proper size
   self.gradInput:resizeAs(input)
   
   -- copy the values of gradInput for each branch into the global gradInput
   for i = 1, self:getNoBranches() do

      self.gradInput:narrow(self.inputDimension, 
                              self.inputOffsets[i], 
                              self.inputSplits[i]
                        ):copy(gradInputsArray[i])
   end

   return self.gradInput
end

function ParallelSeq:__tostring__()
   local tab      = '  '
   local nline    = '\n'
   local next     = '  |`-> '
   local nextlast = '   `-> '
   local ext      = '  |    '
   local extlast  = '       '
   local last     = '  : ==> '
   local str = torch.type(self)
   str = str .. ' {' .. nline .. tab .. 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. nline .. tab .. nextlast .. '(' .. i .. '): ' .. 
                  tostring(self.modules[i]):gsub(nline, nline .. tab .. extlast)
      else
         str = str .. nline .. tab .. next .. '(' .. i .. '): ' .. 
                  tostring(self.modules[i]):gsub(nline, nline .. tab .. ext)
      end
   end
   str = str .. nline .. tab .. last .. 'output'
   str = str .. nline .. '}'
   return str
end