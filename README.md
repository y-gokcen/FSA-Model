# FSA-Model
Repository for running an FSA model in Golang/Emergent

To run this model, steps to run the compcogneuro simulations from https://github.com/emer/leabra are necessary. Once go, leabra, cogentcore, and the simulations are running, then this can be run. It is best to download this in the same folder as the simulations; either in its own folder or within one of the chapters is fine.

On windows, it is run in git bash. Once the file is dowloaded and unziped, in git bash (or the terminal for mac) cd where the FSA code is located (i.e., .../go/leabra/sims/chXYZ/fsa) and type the command "core run". The model should run from there!

Currently there are the "easy" and "hard" tasks for the FSA model. The input represent the presented state at the time, and the output represents the prediction. The rest of the model architecture can be explained from the compcogneuro SIR model.
