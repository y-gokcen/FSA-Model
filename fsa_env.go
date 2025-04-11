// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"


)

// Unlike SIR, there are no actions in FSA ...
// Actions are SIR actions
// type Actions int32 //enums:enum

// const (
// 	Store Actions = iota
// 	Ignore
// 	Recall
// )

// FSAEnv implements the store-ignore-recall task
type FSAEnv struct {
	// name of this environment
	Name string

	// number of different stimuli that can be maintained
	NStim int

	// value for reward, based on whether model output = target
	RewVal float32

	// value for non-reward
	NoRewVal float32

	// Unlike SIR, there are no actions in FSA ...
	// current action
	// Act Actions

	// current stimulus
	Stim int

	// In FSA, there are no requirements on what is maintained ...
	// current stimulus being maintained
	// Maint int

	// In FSA, we need to know the next token for the purposes of
	// of determining targets ...
	NextStim int

	// In FSA, the environment also needs to track what FSA state the
	// environment is in ...
	StateNode int
	NextStateNode int

	// For FSA, specification of the finite state machine ...
	FSAHard bool
	FSARepeatProb float32
	// First, transition probabilities ...
	FSATrans [9][9]float32
	// Second, outputs for each state node ...
	FSAOuts [9]int

	// input pattern with stim
	Input tensor.Float64

	// In FSA, there are no actions ...
	// input pattern with action
	// CtrlInput tensor.Float64

	// output pattern of what to respond
	Output tensor.Float64

	// reward value
	Reward tensor.Float64

	// trial is the step counter within epoch
	Trial env.Counter `view:"inline"`
	
	Sim *Sim
}


func (ev *FSAEnv) Label() string { return ev.Name }

// SetNStim initializes env for given number of stimuli, init states
func (ev *FSAEnv) SetNStim(n int) {
	ev.NStim = n
	ev.Input.SetShape([]int{n})
	// FSA has no control inputs ...
	// ev.CtrlInput.SetShape([]int{int(ActionsN)})
	ev.Output.SetShape([]int{n})
	ev.Reward.SetShape([]int{1})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}
}

func (ev *FSAEnv) SetHardFSA(h bool) {
	ev.FSAHard = h
	ev.FSAOuts[0] = 5 // "F"
	ev.FSAOuts[1] = 0 // "A"
	ev.FSAOuts[2] = 1 // "B"
	ev.FSAOuts[3] = 6 // "G"
	ev.FSAOuts[4] = 6 // "G"
	ev.FSAOuts[7] = 7 // "H"
	ev.FSAOuts[8] = 8 // "I"
	if ev.FSAHard {
		ev.FSAOuts[5] = 2 // "C"
		ev.FSAOuts[6] = 2 // "C"
	} else {
		ev.FSAOuts[5] = 3 // "D"
		ev.FSAOuts[6] = 4 // "E"
	}
}

func (ev *FSAEnv) InitTransProbs(p float32) {
	ev.FSARepeatProb = float32(p)
	// This is a fast way to zero the array, but it garbages ...
	ev.FSATrans = [9][9]float32{}
	ev.FSATrans[0][1] = float32(0.5)
	ev.FSATrans[0][2] = float32(0.5)
	ev.FSATrans[1][3] = float32(ev.FSARepeatProb)
	ev.FSATrans[1][5] = float32(1.0 - ev.FSARepeatProb)
	ev.FSATrans[2][4] = float32(ev.FSARepeatProb)
	ev.FSATrans[2][6] = float32(1.0 - ev.FSARepeatProb)
	ev.FSATrans[3][3] = float32(ev.FSARepeatProb)
	ev.FSATrans[3][5] = float32(1.0 - ev.FSARepeatProb)
	ev.FSATrans[4][4] = float32(ev.FSARepeatProb)
	ev.FSATrans[4][6] = float32(1.0 - ev.FSARepeatProb)
	ev.FSATrans[5][7] = float32(1.0)
	ev.FSATrans[6][8] = float32(1.0)
	ev.FSATrans[7][0] = float32(1.0)
	ev.FSATrans[8][0] = float32(1.0)
}

func (ev *FSAEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
        // In FSA, there is no control input (action) ...		
	// case "CtrlInput":
	// 	return &ev.CtrlInput
	case "Output":
		return &ev.Output
	case "Rew":
		return &ev.Reward
	}
	return nil
}

// StimStr returns a letter string rep of stim (A, B...)
func (ev *FSAEnv) StimStr(stim int) string {
	return string([]byte{byte('A' + stim)})
}

// String returns the current state as a string
func (ev *FSAEnv) String() string {
	return fmt.Sprintf("%s_%s_S%d_rew_%g", ev.StimStr(ev.Stim), ev.StimStr(ev.NextStim), ev.StateNode, ev.Reward.Values[0])
}

func (ev *FSAEnv) Init(run int) {
	ev.Trial.Scale = etime.Trial
	ev.Trial.Init()
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	// There is no Maint field in FSA, unlike SIR ...
	// ev.Maint = -1
	ev.Stim = 7 // the "H" stimulus
	ev.NextStim = 5 // going to restart at "F"
	ev.StateNode = 7 // the "H" state
	ev.NextStateNode = 0 // the "F" state
}

// SetState sets the input, output states
func (ev *FSAEnv) SetState() {
	// Unlike SIR, FSA has no control inputs ...
	// ev.CtrlInput.SetZeros()
	// ev.CtrlInput.Values[ev.Act] = 1
	ev.Input.SetZeros()
	ev.Input.Values[ev.Stim] = 1
	ev.Output.SetZeros()
	ev.Output.Values[ev.NextStim] = 1
}

// SetReward sets reward based on network's output
func (ev *FSAEnv) SetReward(netout int) bool {
	cor := ev.NextStim // already correct
	rw := netout == cor
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

// Step the SIR task
// func (ev *SIREnv) StepSIR() {
// 	for {
// 		ev.Act = Actions(rand.Intn(int(ActionsN)))
// 		if ev.Act == Store && ev.Maint >= 0 { // already full
// 			continue
// 		}
// 		if ev.Act == Recall && ev.Maint < 0 { // nothign
// 			continue
// 		}
// 		break
// 	}
// 	ev.Stim = rand.Intn(ev.NStim)
// 	switch ev.Act {
// 	case Store:
// 		ev.Maint = ev.Stim
// 	case Ignore:
// 	case Recall:
// 		ev.Stim = ev.Maint
// 		ev.Maint = -1
// 	}
// 	ev.SetState()
// }


const (
    PredValid   etime.Modes = iota
    PredError
    PredictedToken
    ValidPct
    Run
)


// GetValidNextTokens returns a map of valid tokens for the current FSA state.
func (ev *FSAEnv) GetValidNextTokens() map[int]bool {
    validTokens := make(map[int]bool)
    for i := 0; i < 9; i++ {
        if ev.FSATrans[ev.StateNode][i] > 0 {
            out := ev.FSAOuts[i]
            validTokens[out] = true
        }
    }
    return validTokens
}


func ArgMax(vals []float64) int {
    maxIdx := 0
    maxVal := vals[0]
    for i, v := range vals {
        if v > maxVal {
            maxVal = v
            maxIdx = i
        }
    }
    return maxIdx
}

func ArgMaxFloat32(vals []float32) int {
	maxIdx := 0
	maxVal := vals[0]
	for i, v := range vals {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}



// LogPrediction logs the prediction results.
func (ev *FSAEnv) LogPrediction(predicted int) {
    if ev.Sim == nil {
        fmt.Println("Warning: ev.Sim is nil in LogPrediction")
        return
    }

    validTokens := ev.GetValidNextTokens()
    isValid := validTokens[predicted]


    if isValid {
        ev.Sim.Stats.SetFloat("PredValid", ev.Sim.Stats.Float("PredValid") + 1.0)
    } else {
        ev.Sim.Stats.SetFloat("PredError", ev.Sim.Stats.Float("PredError") + 1.0)
    }
    total := ev.Sim.Stats.Float("PredValid") + ev.Sim.Stats.Float("PredError")
    if total > 0 {
    	ev.Sim.Stats.SetFloat("ValidPct", ev.Sim.Stats.Float("PredValid") / total)
    }

    ev.Sim.Stats.SetFloat("PredictedToken", float64(predicted))
}

// StepFSA method
func (ev *FSAEnv) StepFSA() {
    ev.Stim = ev.NextStim
    ev.StateNode = ev.NextStateNode

    chosenP := rand.Float32()
    cumulativeP := float32(0.0)
    for i := 0; i < 9; i++ {
        cumulativeP += ev.FSATrans[ev.StateNode][i]
        if chosenP < cumulativeP {
            ev.NextStateNode = i
            break
        }
    }

    ev.NextStim = ev.FSAOuts[ev.NextStateNode]

    //predicted := ev.Sim.LastPred
    
    // Log the prediction
    //ev.LogPrediction(predicted)

    ev.SetState()
}


func (ev *FSAEnv) Step() bool {
	ev.StepFSA()
	ev.Trial.Incr()
	return true
}

func (ev *FSAEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*FSAEnv)(nil)
