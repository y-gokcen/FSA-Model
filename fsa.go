// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// fsa transforms the sir model of dynamic PFC gating into a model of
// serial prediction, with sequences determined by a finite state
// automata (FSA). The model explores the role of PFC gating in long
// distance dependencies in sequence prediction.
package main

//go:generate core generate -add-types

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/styles"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

func main() {
	var (
		nogui       = flag.Bool("nogui", false, "run without GUI")
		hard        = flag.Bool("hard", false, "use hard FSA task")
		repeatProb  = flag.Float64("prob", 0.5, "repeat probability (0.0-1.0)")
		runs        = flag.Int("runs", 10, "number of runs")
		epochs      = flag.Int("epochs", 1000, "epochs per run")
		csvFile     = flag.String("csv", "", "CSV output file (default: auto-generated)")
		testOnly    = flag.Bool("test-only", false, "skip training, only test with loaded weights")
		weightsFile = flag.String("weights", "", "weights file to load for test-only mode")
		testTrials  = flag.Int("test-trials", 10000, "number of test trials")
		runNum        = flag.Int("run", 0, "run number for test-only mode")
		recurrent     = flag.Bool("recurrent", false, "add learnable recurrent Hidden->Hidden connection")
		lesionPFCoutD = flag.Bool("lesion-pfcoutd", false, "lesion PFCoutD neurons via Off flag")
	)
	flag.Parse()

	sim := &Sim{}
	sim.New()

	// Apply command-line parameters
	sim.HardFSA      = *hard
	sim.RepeatProb    = float32(*repeatProb)
	sim.Recurrent     = *recurrent
	sim.LesionPFCoutD = *lesionPFCoutD
	sim.Config.NRuns = *runs
	sim.Config.NEpochs = *epochs
	sim.Config.WeightsFile = *weightsFile

	// Auto-generate CSV filename if not specified
	if *csvFile != "" {
		sim.Config.CSVFile = *csvFile
	} else {
		task := "easy"
		if *hard {
			task = "hard"
		}
		sim.Config.CSVFile = fmt.Sprintf("fsa_%s_p%.2f_r%d.csv",
			task, *repeatProb, *runs)
	}

	sim.ConfigAll()

	// Run with or without GUI
	if *nogui {
		if *testOnly {
			sim.RunTestOnly(*runNum, *testTrials)
		} else {
			sim.RunNoGUI()
		}
	} else {
		sim.RunGUI()
	}
}

// RunTestOnly loads weights and runs testing only (no training)
func (ss *Sim) RunTestOnly(runNum int, numTrials int) {
	fmt.Println("=================================================================")
	fmt.Printf("Test-only mode: Run=%d, TestTrials=%d\n", runNum, numTrials)
	fmt.Printf("Loading weights: %s\n", ss.Config.WeightsFile)
	fmt.Println("=================================================================")

	// Initialize
	ss.Init()
	ss.Stats.SetInt("Run", runNum)

	// Load weights
	if ss.Config.WeightsFile == "" {
		fmt.Println("Error: No weights file specified. Use -weights flag.")
		return
	}

	err := ss.Net.OpenWeightsJSON(core.Filename(ss.Config.WeightsFile))
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}
	fmt.Printf("Loaded weights from: %s\n", ss.Config.WeightsFile)

	// Set up test environment with more trials
	testEnv := ss.Envs.ByMode(etime.Test).(*FSAEnv)
	originalTestTrials := testEnv.Trial.Max
	testEnv.Trial.Max = numTrials

	// Update test loop configuration
	if testStack, ok := ss.Loops.Stacks[etime.Test]; ok {
		testStack.Loops[etime.Trial].Counter.Max = numTrials
	}

	// Set up test CSV filename
	task := "easy"
	if ss.HardFSA {
		task = "hard"
	}
	testFilename := fmt.Sprintf("fsa_%s_p%.2f_run%02d_test_10k.csv",
		task, ss.RepeatProb, runNum+1)

	fmt.Printf("Output file: %s\n", testFilename)

	// Create test CSV
	ss.createTestCSVForRun(testFilename)

	// Run testing
	fmt.Printf("Running %d test trials...\n", numTrials)
	startTime := time.Now()

	ss.TestAll()

	duration := time.Since(startTime)
	fmt.Printf("Testing complete: %v\n", duration)

	// Restore original
	testEnv.Trial.Max = originalTestTrials
	if testStack, ok := ss.Loops.Stacks[etime.Test]; ok {
		testStack.Loops[etime.Trial].Counter.Max = originalTestTrials
	}

	// Close test CSV
	ss.closeTestCSV()

	fmt.Println("=================================================================")
}

// RunNoGUI runs the simulation without GUI for batch processing
func (ss *Sim) RunNoGUI() {
	fmt.Println("=================================================================")
	fmt.Printf("Starting headless run: Hard=%v, RepeatProb=%.2f\n", ss.HardFSA, ss.RepeatProb)
	fmt.Printf("Config: %d runs × %d epochs × %d trials\n",
		ss.Config.NRuns, ss.Config.NEpochs, ss.Config.NTrials)
	fmt.Println("=================================================================")

	startTime := time.Now()

	// Initialize
	ss.Init()

	// Run all training runs using existing loop infrastructure
	ss.Loops.Run(etime.Train)

	trainDuration := time.Since(startTime)
	fmt.Printf("\nTraining complete: %v\n", trainDuration)

	// Close CSV files
	ss.CloseCSV()

	totalDuration := time.Since(startTime)
	fmt.Printf("\nAll complete: %v total\n", totalDuration)
	fmt.Println("=================================================================\n")
}

func (ss *Sim) RunFinalTests() {
	// Temporarily increase test trial count for comprehensive testing
	originalTestTrials := ss.Envs.ByMode(etime.Test).(*FSAEnv).Trial.Max

	// Run more test trials (10x the normal amount = 1000 trials)
	ss.Envs.ByMode(etime.Test).(*FSAEnv).Trial.Max = ss.Config.NTrials * 10

	// Update the test loop configuration
	if testStack, ok := ss.Loops.Stacks[etime.Test]; ok {
		testStack.Loops[etime.Trial].Counter.Max = ss.Config.NTrials * 10
	}

	fmt.Printf("Running %d comprehensive test trials...\n", ss.Config.NTrials*10)

	// Use the existing TestAll() method which properly handles everything
	ss.TestAll()

	// Restore original test trial count
	ss.Envs.ByMode(etime.Test).(*FSAEnv).Trial.Max = originalTestTrials
	if testStack, ok := ss.Loops.Stacks[etime.Test]; ok {
		testStack.Loops[etime.Trial].Counter.Max = originalTestTrials
	}
}

// TestAfterRun runs comprehensive testing after a single run completes
func (ss *Sim) TestAfterRun() {
	currentRun := ss.Stats.Int("Run")

	// Close any existing test CSV
	ss.closeTestCSV()

	// Set up per-run test CSV filename
	task := "easy"
	if ss.HardFSA {
		task = "hard"
	}

	// CRITICAL: Use run%02d format for zero-padding
	testFilename := fmt.Sprintf("fsa_%s_p%.2f_run%02d_test.csv",
		task, ss.RepeatProb, currentRun+1)

	fmt.Printf("  Test file: %s\n", testFilename)

	// Temporarily override the test CSV path
	ss.createTestCSVForRun(testFilename)

	// Run testing with increased trial count
	originalTestTrials := ss.Envs.ByMode(etime.Test).(*FSAEnv).Trial.Max
	ss.Envs.ByMode(etime.Test).(*FSAEnv).Trial.Max = ss.Config.NTrials * 100

	if testStack, ok := ss.Loops.Stacks[etime.Test]; ok {
		testStack.Loops[etime.Trial].Counter.Max = ss.Config.NTrials * 100
	}

	fmt.Printf("  Running %d test trials...\n", ss.Config.NTrials*100)

	ss.TestAll()

	// Restore original
	ss.Envs.ByMode(etime.Test).(*FSAEnv).Trial.Max = originalTestTrials
	if testStack, ok := ss.Loops.Stacks[etime.Test]; ok {
		testStack.Loops[etime.Trial].Counter.Max = originalTestTrials
	}

	// Close this run's test CSV
	ss.closeTestCSV()

	fmt.Printf("Run %d testing complete\n", currentRun)
}

// SaveFinalWeights saves weights for the final run
func (ss *Sim) SaveFinalWeights() {
	// Create filename based on task parameters, not run number
	task := "easy"
	if ss.HardFSA {
		task = "hard"
	}

	filename := fmt.Sprintf("fsa_wts_%s_p%.2f_final.wts.json",
		task, ss.RepeatProb)

	err := ss.Net.SaveWeightsJSON(core.Filename(filename))
	if err != nil {
		fmt.Printf("Error saving final weights: %v\n", err)
	} else {
		fmt.Printf("Saved final weights: %s\n", filename)
	}
}

// SaveRunWeights saves weights after each individual run
func (ss *Sim) SaveRunWeights(run int) {
	task := "easy"
	if ss.HardFSA {
		task = "hard"
	}

	condition := "base"
	if ss.Recurrent && !ss.LesionPFCoutD {
		condition = "recurrent"
	} else if ss.Recurrent && ss.LesionPFCoutD {
		condition = "recurrent_lesion"
	} else if !ss.Recurrent && ss.LesionPFCoutD {
		condition = "base_lesion"
	}
	filename := fmt.Sprintf("fsa_wts_%s_%s_p%.2f_run%02d.wts.json",
		task, condition, ss.RepeatProb, run)

	err := ss.Net.SaveWeightsJSON(core.Filename(filename))
	if err != nil {
		fmt.Printf("Error saving run %d weights: %v\n", run, err)
	} else {
		fmt.Printf("Saved run %d weights: %s\n", run, filename)
	}
}

// ParamSets is the default set of parameters.
// Base is always applied, and others can be optionally
// selected to apply on top of that.
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "no extra learning factors",
			Params: params.Params{
				"Path.Learn.Lrate":       "0.0005", // slower overall is key
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.WtBal.On":    "false",
			}},
		{Sel: "Layer", Desc: "no decay",
			Params: params.Params{
				"Layer.Act.Init.Decay": "0", // key for all layers not otherwise done automatically
			}},
		{Sel: ".BackPath", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2",
			}},
		{Sel: ".BgFixed", Desc: "BG Matrix -> GP wiring",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.8",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
		{Sel: ".RWPath", Desc: "Reward prediction -- into PVi",
			Params: params.Params{
				"Path.Learn.Lrate": "0.02",
				"Path.WtInit.Mean": "0",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
		{Sel: "#Rew", Desc: "Reward layer -- no clamp limits",
			Params: params.Params{
				"Layer.Act.Clamp.Range.Min": "-1",
				"Layer.Act.Clamp.Range.Max": "1",
			}},
		{Sel: ".PFCMntDToOut", Desc: "PFC Deep -> PFC fixed",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.8",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
		{Sel: ".PFCMntDToOut", Desc: "PFC MntD -> PFC Out fixed",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.8",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
		{Sel: ".FmPFCOutD", Desc: "If multiple stripes, PFC OutD needs to be strong b/c avg act says weak",
			Params: params.Params{
				"Path.WtScale.Abs": "1", // increase in proportion to number of stripes
			}},
		{Sel: ".PFCFixed", Desc: "Input -> PFC",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.8",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
		{Sel: ".MatrixPath", Desc: "Matrix learning",
			Params: params.Params{
				"Path.Learn.Lrate":         "0.04", // .04 > .1
				"Path.WtInit.Var":          "0.1",
				"Path.Trace.GateNoGoPosLR": "1",    // 0.1 default
				"Path.Trace.NotGatedLR":    "0.7",  // 0.7 default
				"Path.Trace.Decay":         "1.0",  // 1.0 default
				"Path.Trace.AChDecay":      "0.0",  // not useful even at .1, surprising..
				"Path.Trace.Deriv":         "true", // true default, better than false
			}},
		{Sel: ".MatrixLayer", Desc: "exploring these options",
			Params: params.Params{
				"Layer.Act.XX1.Gain":       "100",
				"Layer.Inhib.Layer.Gi":     "1.9",
				"Layer.Inhib.Layer.FB":     "0.5",
				"Layer.Inhib.Pool.On":      "true",
				"Layer.Inhib.Pool.Gi":      "2.1", // def 1.9
				"Layer.Inhib.Pool.FB":      "0",
				"Layer.Inhib.Self.On":      "true",
				"Layer.Inhib.Self.Gi":      "0.4", // def 0.3
				"Layer.Inhib.ActAvg.Init":  "0.05",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: "#GPiThal", Desc: "defaults also set automatically by layer but included here just to be sure",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":     "1.8",
				"Layer.Inhib.Layer.FB":     "1", // was 0.5
				"Layer.Inhib.Pool.On":      "false",
				"Layer.Inhib.ActAvg.Init":  ".2",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Act.Dt.GTau":        "3",
				"Layer.GPiGate.GeGain":     "3",
				"Layer.GPiGate.NoGo":       "1",   // 1.25?
				"Layer.GPiGate.Thr":        "0.2", // 0.25?
			}},
		{Sel: "#GPeNoGo", Desc: "GPe is a regular layer -- needs special params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":     "2.4", // 2.4 > 2.2 > 1.8
				"Layer.Inhib.Layer.FB":     "0.5",
				"Layer.Inhib.Layer.FBTau":  "3", // otherwise a bit jumpy
				"Layer.Inhib.Pool.On":      "false",
				"Layer.Inhib.ActAvg.Init":  ".2",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: ".PFC", Desc: "pfc defaults",
			Params: params.Params{
				"Layer.Inhib.Layer.On":     "false",
				"Layer.Inhib.Pool.On":      "true",
				"Layer.Inhib.Pool.Gi":      "1.8",
				"Layer.Inhib.Pool.FB":      "1",
				"Layer.Inhib.ActAvg.Init":  "0.2",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: "#Input", Desc: "Basic params",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init":  "0.25",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: "#Output", Desc: "Basic params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":     "2",
				"Layer.Inhib.Layer.FB":     "0.5",
				"Layer.Inhib.ActAvg.Init":  "0.25",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		// Unlike SIR, there are no input to output connections ...
		// {Sel: "#InputToOutput", Desc: "weaker",
		// 	Params: params.Params{
		// 		"Path.WtScale.Rel": "0.5",
		// 	}},
		{Sel: "#Hidden", Desc: "Basic params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2",
				"Layer.Inhib.Layer.FB": "0.5",
			}},
		{Sel: "#SNc", Desc: "allow negative",
			Params: params.Params{
				"Layer.Act.Clamp.Range.Min": "-1",
				"Layer.Act.Clamp.Range.Max": "1",
			}},
		{Sel: "#RWPred", Desc: "keep it guessing",
			Params: params.Params{
				"Layer.RW.PredRange.Min": "0.01", // increasing to .05, .95 can be useful for harder tasks
				"Layer.RW.PredRange.Max": "0.99",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"10" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"1000"`

	// total number of trials per epochs per run
	NTrials int `default:"100"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"5"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"-1"`

	// CSVFile is an optional path for writing per-trial prediction statistics.
	// If empty, no CSV is written. Relative paths are relative to the working directory.
	CSVFile string `default:""`

	WeightsFile string `default:""`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// HardFSA is a boolean -- use the hard FSA if true
	HardFSA bool

	// RepeatProb is the probability of repetition in the middle of
	// the FSA generated sequence
	RepeatProb float32

	// BurstDaGain is the strength of dopamine bursts: 1 default -- reduce for PD OFF, increase for PD ON
	BurstDaGain float32

	// DipDaGain is the strength of dopamine dips: 1 default -- reduce to siulate D2 agonists
	DipDaGain float32

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`

	LastPred      int  // stores the most recent model prediction
	Recurrent     bool // add learnable recurrent Hidden->Hidden connection
	LesionPFCoutD bool // lesion PFCoutD via LesionNeurons (sets Off flag)

	// CSV logging state (not displayed in GUI)
	csvFile          *os.File    `display:"-"`
	csvWriter        *csv.Writer `display:"-"`
	csvHeaderWrote   bool        `display:"-"`
	csvClosed        bool        `display:"-"`
	csvTestWriter    *csv.Writer
	csvTestFile      *os.File
	csvTestLastRun   int
	csvTestLastEpoch int
	csvTestLastTrial int

	// last CSV write coordinates to avoid duplicate rows per trial
	csvLastRun   int `display:"-"`
	csvLastEpoch int `display:"-"`
	csvLastTrial int `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	ss.Net = leabra.NewNetwork("FSA")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Stats.SetInt("Expt", 0)
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
	// init CSV last-write trackers so first trial logs
	ss.csvLastRun, ss.csvLastEpoch, ss.csvLastTrial = -1, -1, -1
}

func (ss *Sim) Defaults() {
	ss.BurstDaGain = 1
	ss.DipDaGain = 1
	ss.HardFSA = false
	ss.RepeatProb = 0.5
}

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *FSAEnv
	if len(ss.Envs) == 0 {
		trn = &FSAEnv{Sim: ss} // Assign Sim reference
		tst = &FSAEnv{Sim: ss}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*FSAEnv)
		tst = ss.Envs.ByMode(etime.Test).(*FSAEnv)
	}

	trn.Name = etime.Train.String()
	trn.SetNStim(9)
	trn.RewVal = 1
	trn.NoRewVal = 0
	trn.Trial.Max = ss.Config.NTrials

	tst.Name = etime.Test.String()
	tst.SetNStim(9)
	tst.RewVal = 1
	tst.NoRewVal = 0
	tst.Trial.Max = ss.Config.NTrials

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	//net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	rew, rp, da := net.AddRWLayers("", 2)
	da.Name = "SNc"

	// number of tokens from 7 to 9 ...
	inp := net.AddLayer2D("Input", 1, 9, leabra.InputLayer)
	// There is no control input in the FSA task ...
	// ctrl := net.AddLayer2D("CtrlInput", 1, 3, leabra.InputLayer)
	out := net.AddLayer2D("Output", 1, 9, leabra.TargetLayer)
	// match size to output ...
	hid := net.AddLayer2D("Hidden", 9, 9, leabra.SuperLayer)

	// args: nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX
	mtxGo, mtxNoGo, gpe, gpi, cin, pfcMnt, pfcMntD, pfcOut, pfcOutD := net.AddPBWM("", 1, 1, 1, 1, 9, 1, 9) //used to be 1111319
	_ = gpe
	_ = gpi
	_ = pfcMnt
	_ = pfcMntD
	_ = pfcOut
	_ = cin

	cin.CIN.RewLays.Add(rew.Name, rp.Name)

	inp.PlaceAbove(rew)
	out.PlaceRightOf(inp, 2)
	// ctrl.PlaceBehind(inp, 2)
	// hid.PlaceBehind(ctrl, 2)
	hid.PlaceBehind(inp, 2)
	mtxGo.PlaceRightOf(rew, 2)
	pfcMnt.PlaceRightOf(out, 2)

	full := paths.NewFull()
	fmin := paths.NewRect()
	fmin.Size.Set(1, 1)
	fmin.Scale.Set(1, 1)
	fmin.Wrap = true

	// net.ConnectLayers(ctrl, rp, full, leabra.RWPath)
	net.ConnectLayers(pfcMntD, rp, full, leabra.RWPath)
	net.ConnectLayers(pfcOutD, rp, full, leabra.RWPath)

	// net.ConnectLayers(ctrl, mtxGo, fmin, leabra.MatrixPath)
	// net.ConnectLayers(ctrl, mtxNoGo, fmin, leabra.MatrixPath)
	// FSA has no control inputs, but something needs to drive
	// the striatum layers. Let's try driving them from input ...
	// net.ConnectLayers(ctrl, mtxGo, fmin, leabra.MatrixPath)
	// net.ConnectLayers(ctrl, mtxNoGo, fmin, leabra.MatrixPath)
	net.ConnectLayers(inp, mtxGo, fmin, leabra.MatrixPath)
	net.ConnectLayers(inp, mtxNoGo, fmin, leabra.MatrixPath)
	pt := net.ConnectLayers(inp, pfcMnt, fmin, leabra.ForwardPath)
	pt.AddClass("PFCFixed")

	net.ConnectLayers(inp, hid, full, leabra.ForwardPath)
	net.BidirConnectLayers(hid, out, full)
	// Add recurrent Hidden->Hidden connection if flag set
	if ss.Recurrent {
		net.ConnectLayers(hid, hid, full, leabra.ForwardPath)
	}
	pt = net.ConnectLayers(pfcOutD, hid, full, leabra.ForwardPath)
	pt.AddClass("FmPFCOutD")
	pt = net.ConnectLayers(pfcOutD, out, full, leabra.ForwardPath)
	pt.AddClass("FmPFCOutD")
	// In the FSA task, it doesn't make sense to have direct connections
	// from the input layer to the output layer ...
	// net.ConnectLayers(inp, out, full, leabra.ForwardPath)

	net.Build()
	net.Defaults()

	da.AddAllSendToBut() // send dopamine to all layers..
	gpi.SendPBWMParams()

	ss.ApplyParams()
	net.InitWeights()

	// Set lower weights for a specific MatrixGo unit (e.g., unit 0)
	ss.SetUnitWeightsEmer(6, 0.05, "MatrixGo")  // Stripe 0, Token G
	ss.SetUnitWeightsEmer(15, 0.05, "MatrixGo") // Stripe 1, Token G
	ss.SetUnitWeightsEmer(6, 0.90, "MatrixNoGo")
	ss.SetUnitWeightsEmer(15, 0.90, "MatrixNoGo")

	// NEW: Help A in both stripes equally
	ss.SetUnitWeightsEmer(0, 0.80, "MatrixGo") // Stripe 0, Token A
	ss.SetUnitWeightsEmer(9, 0.80, "MatrixGo") // Stripe 1, Token A

	// NEW: Help B in both stripes equally
	ss.SetUnitWeightsEmer(1, 0.80, "MatrixGo")  // Stripe 0, Token B
	ss.SetUnitWeightsEmer(10, 0.80, "MatrixGo") // Stripe 1, Token B
}

// SetMatrixGoUnitWeightsEmer demonstrates using emer.Layer interface to edit MatrixGo unit weights.
func (ss *Sim) SetUnitWeightsEmer(unitIdx int, wt float32, layerName string) {
	var ly emer.Layer = ss.Net.LayerByName(layerName)
	// Type assert to *leabra.Layer to access projections and synapses.
	lb := ly.(*leabra.Layer)
	// Use the emer.Layer interface methods to iterate incoming projections.
	nRecv := lb.NumRecvPaths()
	for pi := 0; pi < nRecv; pi++ {
		prj := lb.RecvPath(pi).(*leabra.Path)
		sy := prj.Syns
		if sy == nil {
			continue
		}
		nSend := prj.Send.Shape.Len()
		for sendIdx := 0; sendIdx < nSend; sendIdx++ {
			syIdx := unitIdx*nSend + sendIdx
			if syIdx < len(sy) {
				sy[syIdx].Wt = wt
			}
		}
	}
}

func (ss *Sim) ApplyParams() {
	var trn *FSAEnv
	var tst *FSAEnv

	if ss.Loops != nil {
		trn := ss.Loops.Stacks[etime.Train]
		trn.Loops[etime.Run].Counter.Max = ss.Config.NRuns
		trn.Loops[etime.Epoch].Counter.Max = ss.Config.NEpochs
	}
	ss.Params.SetAll()

	trn = ss.Envs.ByMode(etime.Train).(*FSAEnv)
	tst = ss.Envs.ByMode(etime.Test).(*FSAEnv)
	trn.SetHardFSA(ss.HardFSA)
	trn.InitTransProbs(ss.RepeatProb)
	tst.SetHardFSA(ss.HardFSA)
	tst.InitTransProbs(ss.RepeatProb)

	matg := ss.Net.LayerByName("MatrixGo")
	matn := ss.Net.LayerByName("MatrixNoGo")

	matg.Matrix.BurstGain = ss.BurstDaGain
	matg.Matrix.DipGain = ss.DipDaGain
	matn.Matrix.BurstGain = ss.BurstDaGain
	matn.Matrix.DipGain = ss.DipDaGain
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	ss.GUI.StopNow = false
	ss.ViewUpdate.View = nil
	ss.ApplyParams()
	ss.NewRun()
	// ss.ViewUpdate.RecordSyns() // disabled for headless operation
	// ss.ViewUpdate.Update()     // disabled for headless operation
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.Config.NTrials

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			// NEW: Clear activations for non-PFC layers before applying inputs
			net := ss.Net
			for _, ly := range net.Layers {
				// Don't clear PFC layers (they should maintain)
				if strings.Contains(ly.Name, "PFC") {
					continue
				}
				// Don't clear BG/thalamus layers (gating mechanism)
				if strings.Contains(ly.Name, "Matrix") ||
					strings.Contains(ly.Name, "GP") ||
					strings.Contains(ly.Name, "SNc") ||
					ly.Name == "RWPred" {
					continue
				}
				// Don't clear Hidden in recurrent mode — needs cross-trial memory
				if ss.Recurrent && ly.Name == "Hidden" {
					continue
				}
				// Clear Hidden, Input, Output
				ly.InitActs()
			}

			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunDone", func() {
		currentRun := ss.Stats.Int("Run")

		// Save weights after this run
		fmt.Printf("\n[Run %d] Saving weights...\n", currentRun)
		ss.SaveRunWeights(currentRun)

		// Close this run's training CSV
		ss.CloseTrainingCSV()

		// Test after each run completes
		fmt.Printf("[Run %d] Testing...\n", currentRun)
		ss.TestAfterRun()

		if currentRun >= ss.Config.NRuns-1 {
			expt := ss.Stats.Int("Expt")
			ss.Stats.SetInt("Expt", expt+1)
		}
	})
	// Add phase hooks for BOTH Train and Test stacks
	for mode := range ls.Stacks {
		stack := ls.Stacks[mode]

		if mode == etime.Train {
			stack.Loops[etime.Epoch].OnStart.Add("ResetEpochPredStats", func() {
				ss.Stats.SetFloat("EpochValid", 0.0)
				ss.Stats.SetFloat("EpochError", 0.0)
				ss.Stats.SetFloat("EpochValidPct", 0.0)
			})
		}

		cyc, _ := stack.Loops[etime.Cycle]
		plus := cyc.EventByName("MinusPhase:End")

		plus.OnEvent.InsertBefore("MinusPhase:End", "ApplyReward", func() bool {
			// Use correct mode for reward/prediction
			isTrain := ss.Context.Mode == etime.Train
			ss.ApplyReward(isTrain)
			ss.Envs.ByMode(ss.Context.Mode).(*FSAEnv).LogPrediction(ss.LastPred)

			// Only log PFC activation during training (optional)
			if ss.Context.Mode == etime.Train {
				if ly := ss.Net.LayerByName("PFCmnt"); ly != nil {
					units := ly.Neurons
					var sum float64
					for i := range units {
						sum += float64(units[i].Act)
					}
					meanAct := sum / float64(len(units))
					ss.Stats.SetFloat("PFCmntActM", meanAct)
				}
			}

			return true
		})
	}

	// Train stop early condition
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("NZeroStop", func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Config.NZero
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	})

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})
	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	////////////////////////////////////////////
	// GUI

	// GUI updates disabled for headless/HPC operation
	// leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	// leabra.LooperUpdatePlots(ls, &ss.GUI)
	// ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	// ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*FSAEnv)
	ev.Step()

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.String())
	for _, lnm := range lays {
		if lnm == "Rew" {
			continue
		}
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// ApplyReward computes reward based on network output and applies it.
// Call at start of 3rd quarter (plus phase).
func (ss *Sim) ApplyReward(train bool) {
	var en *FSAEnv
	if train {
		en = ss.Envs.ByMode(etime.Train).(*FSAEnv)
	} else {
		en = ss.Envs.ByMode(etime.Test).(*FSAEnv)
	}
	// Unlike SIR, there is the possibility of reward on every trial ...
	// if en.Act != Recall { // only reward on recall trials!
	// 	return
	// }
	out := ss.Net.LayerByName("Output")
	mxi := out.Pools[0].Inhib.Act.MaxIndex
	ss.LastPred = int(mxi)
	en.SetReward(int(mxi))
	pats := en.State("Rew")
	ly := ss.Net.LayerByName("Rew")
	ly.ApplyExt1DTsr(pats)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	currentRun := ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur

	ss.InitRandSeed(currentRun)
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	ss.ApplyParams()
	// Apply functional lesion: sets Off flag on all neurons in target layer
	if ss.LesionPFCoutD {
		ly := ss.Net.LayerByName("PFCoutD")
		ly.LesionNeurons(1.0)
		fmt.Println("PFCoutD lesioned (100% neurons silenced via Off flag)")
	}
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)

	// NEW: Close previous run's training CSV
	ss.CloseTrainingCSV()

	// NEW: Create per-run training CSV filename
	task := "easy"
	if ss.HardFSA {
		task = "hard"
	}
	condition := "base"
	if ss.Recurrent && !ss.LesionPFCoutD {
		condition = "recurrent"
	} else if ss.Recurrent && ss.LesionPFCoutD {
		condition = "recurrent_lesion"
	} else if !ss.Recurrent && ss.LesionPFCoutD {
		condition = "base_lesion"
	}
	ss.Config.CSVFile = fmt.Sprintf("fsa_%s_%s_p%.2f_run%02d.csv",
		task, condition, ss.RepeatProb, currentRun+1)

	ss.InitCSV() // open this run's CSV

	// reset CSV last-write trackers at new run boundary
	ss.csvLastRun, ss.csvLastEpoch, ss.csvLastTrial = -1, -1, -1
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetFloat("DA", 0.0)
	ss.Stats.SetFloat("AbsDA", 0.0)
	ss.Stats.SetFloat("RewPred", 0.0)
	ss.Stats.SetFloat("PredValid", 0.0)       // Track valid predictions
	ss.Stats.SetFloat("PredError", 0.0)       // Track invalid predictions
	ss.Stats.SetFloat("PredictedToken", -1.0) // Track last predicted token
	ss.Stats.SetFloat("ValidPct", 0.0)
	ss.Stats.SetFloat("EpochValid", 0.0)
	ss.Stats.SetFloat("EpochError", 0.0)
	ss.Stats.SetFloat("EpochValidPct", 0.0)
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("PFCmntToken", "")
	ss.Stats.SetString("PFCmntDToken", "")
	ss.Stats.SetString("PFCoutDToken", "")

	ss.Logs.InitErrStats() // Initialize error tracking
}

func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)

	// Keep running total of valid/invalid predictions
	ss.Stats.SetFloat("PredValid", ss.Stats.Float("PredValid"))
	ss.Stats.SetFloat("PredError", ss.Stats.Float("PredError"))
	ss.Stats.SetFloat("PredictedToken", ss.Stats.Float("PredictedToken"))
	ss.Stats.SetFloat("ValidPct", ss.Stats.Float("ValidPct"))
	ss.Stats.SetFloat("EpochValid", ss.Stats.Float("EpochValid"))
	ss.Stats.SetFloat("EpochError", ss.Stats.Float("EpochError"))
	ss.Stats.SetFloat("EpochValidPct", ss.Stats.Float("EpochValidPct"))

	// always use training epoch
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	params := fmt.Sprintf("hard: %t, prob: %g, burst: %g, dip: %g", ss.HardFSA, ss.RepeatProb, ss.BurstDaGain, ss.DipDaGain)
	ss.Stats.SetString("RunName", params)

	out := ss.Net.LayerByName("Output")

	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}

	// --- Log PFC maintenance tokens --- //
	pfcM := ss.Net.LayerByName("PFCmnt")
	pfcMD := ss.Net.LayerByName("PFCmntD")

	// Find max activation index from ActM of PFCmnt
	pfcMIdx := 0
	maxAct := float32(-1)
	for i := range pfcM.Neurons {
		if pfcM.Neurons[i].ActM > maxAct {
			maxAct = pfcM.Neurons[i].ActM
			pfcMIdx = i
		}
	}

	// Map to token letter A–I
	if pfcMIdx >= 0 && pfcMIdx < 9 {
		ss.Stats.SetString("PFCmntToken", string(rune('A'+pfcMIdx)))
	} else {
		ss.Stats.SetString("PFCmntToken", "")
	}

	// --- Do the same for PFCmntD (optional, recommended) ---
	pfcMDIdx := 0
	maxActD := float32(-1)
	for i := range pfcMD.Neurons {
		if pfcMD.Neurons[i].ActM > maxActD {
			maxActD = pfcMD.Neurons[i].ActM
			pfcMDIdx = i
		}
	}

	if pfcMDIdx >= 0 && pfcMDIdx < 9 {
		ss.Stats.SetString("PFCmntDToken", string(rune('A'+pfcMDIdx)))
	} else {
		ss.Stats.SetString("PFCmntDToken", "")
	}

	// --- Log PFCoutD token ---
	pfcOutD := ss.Net.LayerByName("PFCoutD")

	pfcOutDIdx := 0
	maxActOutD := float32(-1)
	for i := range pfcOutD.Neurons {
		if pfcOutD.Neurons[i].ActM > maxActOutD {
			maxActOutD = pfcOutD.Neurons[i].ActM
			pfcOutDIdx = i
		}
	}

	if pfcOutDIdx >= 0 && pfcOutDIdx < 9 {
		ss.Stats.SetString("PFCoutDToken", string(rune('A'+pfcOutDIdx)))
	} else {
		ss.Stats.SetString("PFCoutDToken", "")
	}

	snc := ss.Net.LayerByName("SNc")
	ss.Stats.SetFloat32("DA", snc.Neurons[0].Act)
	ss.Stats.SetFloat32("AbsDA", math32.Abs(snc.Neurons[0].Act))
	rp := ss.Net.LayerByName("RWPred")
	ss.Stats.SetFloat32("RewPred", rp.Neurons[0].Act)

}

//////////////////////////////////////////////////////////////////////
// 		Logging

// ConfigLogs sets up logging configuration, including the new prediction log.
func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "Expt")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	//ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "PredValid")
	//ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "PredError")
	//ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "PredictedToken")
	//ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "ValidPct")

	ss.Logs.AddStatAggItem("PredValid", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PredError", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PredictedToken", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ValidPct", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("EpochValid", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("EpochError", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("EpochValidPct", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("DA", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AbsDA", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("RewPred", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatStringItem(etime.Train, etime.Trial, "PFCmntToken")
	ss.Logs.AddStatStringItem(etime.Train, etime.Trial, "PFCmntDToken")
	ss.Logs.AddStatStringItem(etime.Train, etime.Trial, "PFCoutDToken")

	ss.Logs.CreateTables()

	ss.Logs.PlotItems("PctErr", "AbsDA", "RewPred", "ValidPct", "EpochValidPct") // Add PredValid to plots
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	// Auto-save CSV if user pressed Stop (GUI sets StopNow) after trial logging
	if time == etime.Trial && false { // ss.GUI.StopNow disabled for headless
		fmt.Println("[CSV] Stop pressed: closing CSV file.")
		ss.CloseCSV()
	}

	// Fallback: if row somehow not written during phase hook, write at end of trial
	// Log CSV for both Train and Test modes
	if time == etime.Trial {
		ss.LogTrialCSV()
	}

	if mode == etime.Test {
		// ss.GUI.UpdateTableView(etime.Test, etime.Trial) // disabled for headless operation
	}
}

//////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.Options.LayerNameSize = 0.03

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.85, 2.25)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{X: 0, Y: 0, Z: 0}, math32.Vector3{X: 0, Y: 1, Z: 0})

	labs := []string{"  A B C D E F G H I", " A B C D E F G H I", " A B C D E F G H I ", "A B C D E F G H I", "A B C D E F G H I  ", "  A B C D E F G H I  "}
	nv.ConfigLabels(labs)

	lays := []string{"Input", "PFCmnt", "PFCmntD", "PFCout", "PFCoutD", "Output"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .08
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(math32.Vec3(1, 0.3, 0.5))
		lbl.Styles.Text.WhiteSpace = styles.WhiteSpacePre
	}
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "FSA"
	ss.GUI.MakeBody(ss, "fsa", title, `fsa transforms the sir model of dynamic PFC gating into a model of serial prediction, with sequences determined by a finite state automata (FSA). The model explores the role of PFC gating in long distance dependencies in sequence prediction.`)
	ss.GUI.CycleUpdateInterval = 10000

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.003
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	ss.ConfigNetView(nv)
	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Test, etime.Trial)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "Save Weights",
		Icon:    icons.Save,
		Tooltip: "Save network weights",
		Active:  egui.ActiveAlways,
		Func: func() {
			fn := fmt.Sprintf("fsa_wts_run%d_epc%d.wts.json",
				ss.Stats.Int("Run"), ss.Stats.Int("Epoch"))
			err := ss.Net.SaveWeightsJSON(core.Filename(fn))
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Saved: %s\n", fn)
			}
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "Open Weights",
		Icon:    icons.Open,
		Tooltip: "Load weights (set Config.WeightsFile first)",
		Active:  egui.ActiveAlways,
		Func: func() {
			if ss.Config.WeightsFile != "" {
				err := ss.Net.OpenWeightsJSON(core.Filename(ss.Config.WeightsFile))
				if err != nil {
					fmt.Printf("Error: %v\n", err)
				} else {
					fmt.Printf("Loaded: %s\n", ss.Config.WeightsFile)
					ss.GUI.UpdateNetView()
				}
			}
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:  "Show Methods",
		Icon:   icons.Info,
		Active: egui.ActiveAlways,
		Func: func() {
			t := reflect.TypeOf(ss.Net)
			fmt.Println("\n=== Network Methods ===")
			for i := 0; i < t.NumMethod(); i++ {
				m := t.Method(i)
				name := strings.ToLower(m.Name)
				if strings.Contains(name, "wt") ||
					strings.Contains(name, "save") ||
					strings.Contains(name, "write") {
					fmt.Printf("  %s\n", m.Name)
				}
			}
			fmt.Println("=====================\n")
		},
	})

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch9/sir/README.md")
		},
	})
	// Manual CSV save button
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Save CSV Now",
		Icon:    icons.Save,
		Tooltip: "Flush and close the currently configured CSV file immediately.",
		Active:  egui.ActiveAlways,
		Func: func() {
			if ss.csvWriter != nil {
				fmt.Println("[CSV] Manual save requested: closing CSV file.")
			}
			ss.CloseCSV()
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

// ===================== CSV Logging Support =====================

// InitCSV opens the CSV file for appending if configured. Writes header once.
func (ss *Sim) InitCSV() {
	if ss.Config.CSVFile == "" { // provide default filename if none supplied
		ss.Config.CSVFile = "fsa_trials.csv"
	}
	if ss.csvFile != nil && !ss.csvClosed { // already open
		return
	}
	// Open (or create) file in append mode
	path := ss.Config.CSVFile
	// If target file exists with a wide schema header, switch to *_letters.csv to avoid mixing
	if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
		rf, rerr := os.Open(path)
		if rerr == nil {
			r := csv.NewReader(rf)
			r.FieldsPerRecord = -1
			if rec, rerr2 := r.Read(); rerr2 == nil {
				// Wide header typically contains Stim_A as a field
				isWide := len(rec) >= 10 && containsField(rec, "Stim_A")
				if isWide {
					newPath := ensureLettersCSVPath(path)
					fmt.Printf("[CSV] Existing file looks wide-format; writing letters format to %s\n", newPath)
					path = newPath
					ss.Config.CSVFile = newPath
				}
			}
			_ = rf.Close()
		}
	}
	if !filepath.IsAbs(path) {
		// leave as relative; user can control working dir
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Printf("CSV open error: %v\n", err)
		return
	}
	ss.csvFile = f
	ss.csvWriter = csv.NewWriter(f)
	fi, _ := f.Stat()
	if fi != nil && fi.Size() == 0 { // new file
		ss.writeCSVHeader()
	}
	ss.csvClosed = false
	// ensure last-write trackers won't suppress first row
	if ss.csvLastRun == 0 && ss.csvLastEpoch == 0 && ss.csvLastTrial == 0 {
		ss.csvLastRun, ss.csvLastEpoch, ss.csvLastTrial = -1, -1, -1
	}
}

func (ss *Sim) writeCSVHeader() {
	if ss.csvWriter == nil || ss.csvHeaderWrote {
		return
	}
	hdr := []string{"Run", "Epoch", "Trial", "StateNode", "Stim", "NextStim",
		"Predicted", "Valid", "Reward", "DA", "AbsDA", "RewPred",
		"ValidPct", "EpochValidPct", "PFCmntToken", "PFCmntDToken", "PFCoutDToken"}

	if err := ss.csvWriter.Write(hdr); err == nil {
		ss.csvWriter.Flush()
		ss.csvHeaderWrote = true
	}
}

// LogTrialCSV writes a single per-trial row for both Train and Test modes
func (ss *Sim) LogTrialCSV() {
	ctx := &ss.Context

	// Only log for Train and Test modes
	if ctx.Mode != etime.Train && ctx.Mode != etime.Test {
		return
	}

	// Determine which CSV file to use based on mode
	var writer *csv.Writer
	var lastRun, lastEpoch, lastTrial *int
	var env *FSAEnv

	if ctx.Mode == etime.Train {
		// Training mode - use existing CSV
		if ss.csvWriter == nil {
			ss.InitCSV()
			if ss.csvWriter == nil {
				return
			}
		}
		writer = ss.csvWriter
		lastRun = &ss.csvLastRun
		lastEpoch = &ss.csvLastEpoch
		lastTrial = &ss.csvLastTrial
		env = ss.Envs.ByMode(etime.Train).(*FSAEnv)

	} else { // Test mode
		// For test mode, create a separate CSV file
		if ss.csvWriter == nil {
			ss.InitCSV() // Initialize training CSV first
		}

		// Use a separate test CSV - lazy initialization
		if ss.csvTestWriter == nil {
			ss.initTestCSVSimple()
		}

		if ss.csvTestWriter == nil {
			return // Failed to initialize
		}

		writer = ss.csvTestWriter
		lastRun = &ss.csvTestLastRun
		lastEpoch = &ss.csvTestLastEpoch
		lastTrial = &ss.csvTestLastTrial
		env = ss.Envs.ByMode(etime.Test).(*FSAEnv)
	}

	// Guard against duplicate writes
	curRun := ss.Stats.Int("Run")
	curEpoch := ss.Stats.Int("Epoch")
	curTrial := ss.Stats.Int("Trial")
	if *lastRun == curRun && *lastEpoch == curEpoch && *lastTrial == curTrial {
		return
	}

	// Build the row
	predicted := ss.LastPred
	target := env.NextStim
	validTokens := env.GetValidNextTokens()
	_, isValid := validTokens[predicted]
	reward := env.Reward.Values[0]

	row := []string{
		strconv.Itoa(curRun),
		strconv.Itoa(curEpoch),
		strconv.Itoa(curTrial),
		strconv.Itoa(env.StateNode),
		env.StimStr(env.Stim),
		env.StimStr(target),
		env.StimStr(predicted),
		boolToStr(isValid),
		fmt.Sprintf("%g", reward),
		fmt.Sprintf("%.5f", ss.Stats.Float("DA")),
		fmt.Sprintf("%.5f", ss.Stats.Float("AbsDA")),
		fmt.Sprintf("%.5f", ss.Stats.Float("RewPred")),
		fmt.Sprintf("%.5f", ss.Stats.Float("ValidPct")),
		fmt.Sprintf("%.5f", ss.Stats.Float("EpochValidPct")),
		ss.Stats.String("PFCmntToken"),
		ss.Stats.String("PFCmntDToken"),
		ss.Stats.String("PFCoutDToken"),
	}

	if err := writer.Write(row); err != nil {
		fmt.Printf("CSV write error: %v\n", err)
	}
	writer.Flush()
	*lastRun, *lastEpoch, *lastTrial = curRun, curEpoch, curTrial
}

// containsField returns true if any field case-sensitively equals s.
func containsField(rec []string, s string) bool {
	for _, f := range rec {
		if strings.TrimSpace(f) == s {
			return true
		}
	}
	return false
}

// ensureLettersCSVPath returns a filename with _letters suffix before extension,
// ensuring we don't overwrite existing files by adding a numeric suffix if needed.
func ensureLettersCSVPath(path string) string {
	dir := filepath.Dir(path)
	base := filepath.Base(path)
	ext := filepath.Ext(base)
	name := strings.TrimSuffix(base, ext)
	cand := filepath.Join(dir, name+"_letters"+ext)
	if _, err := os.Stat(cand); os.IsNotExist(err) {
		return cand
	}
	for i := 1; i < 1000; i++ {
		c := filepath.Join(dir, fmt.Sprintf("%s_letters_%d%s", name, i, ext))
		if _, err := os.Stat(c); os.IsNotExist(err) {
			return c
		}
	}
	return cand
}

func (ss *Sim) CloseCSV() {
	// Close training CSV
	ss.CloseTrainingCSV()
	// Close test CSV
	ss.closeTestCSV()
}

// CloseTrainingCSV closes only the training CSV (not test CSV)
func (ss *Sim) CloseTrainingCSV() {
	if ss.csvFile != nil && !ss.csvClosed {
		ss.csvWriter.Flush()
		_ = ss.csvFile.Close()
		ss.csvFile = nil
		ss.csvWriter = nil
		ss.csvClosed = true
	}
}

// initTestCSVSimple creates a test CSV file with minimal complexity
func (ss *Sim) initTestCSVSimple() {
	// Generate test CSV filename from training CSV filename
	trainPath := ss.Config.CSVFile
	if trainPath == "" {
		trainPath = "fsa_trials.csv"
	}

	testPath := strings.Replace(trainPath, ".csv", "_test.csv", 1)
	if testPath == trainPath {
		testPath = trainPath + "_test.csv"
	}

	// Open file for appending
	f, err := os.OpenFile(testPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Printf("Test CSV open error: %v\n", err)
		return
	}

	ss.csvTestFile = f
	ss.csvTestWriter = csv.NewWriter(f)

	// Write header if file is new
	fi, _ := f.Stat()
	if fi != nil && fi.Size() == 0 {
		hdr := []string{"Run", "Epoch", "Trial", "StateNode", "Stim", "NextStim",
			"Predicted", "Valid", "Reward", "DA", "AbsDA", "RewPred",
			"ValidPct", "EpochValidPct", "PFCmntToken", "PFCmntDToken", "PFCoutDToken"}
		ss.csvTestWriter.Write(hdr)
		ss.csvTestWriter.Flush()
	}

	// Initialize guards
	ss.csvTestLastRun = -1
	ss.csvTestLastEpoch = -1
	ss.csvTestLastTrial = -1

	fmt.Printf("[CSV] Test logging enabled: %s\n", testPath)
}

// createTestCSVForRun creates a fresh test CSV for a specific run
func (ss *Sim) createTestCSVForRun(filename string) {
	// Close any existing test CSV
	ss.closeTestCSV()

	// Open new file (overwrite if exists - each run gets fresh file)
	f, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Test CSV create error: %v\n", err)
		return
	}

	ss.csvTestFile = f
	ss.csvTestWriter = csv.NewWriter(f)

	// Write header
	hdr := []string{"Run", "Epoch", "Trial", "Sequence", "StateNode", "Input",
		"Output", "Target", "Reward",
		"PFCmntToken", "PFCmntDToken", "PFCoutDToken"}

	if err := ss.csvTestWriter.Write(hdr); err != nil {
		fmt.Printf("Test CSV header error: %v\n", err)
	}
	ss.csvTestWriter.Flush()

	// Reset guards
	ss.csvTestLastRun = -1
	ss.csvTestLastEpoch = -1
	ss.csvTestLastTrial = -1
}

// closeTestCSV closes the test CSV if open
func (ss *Sim) closeTestCSV() {
	if ss.csvTestWriter != nil {
		ss.csvTestWriter.Flush()
		if ss.csvTestFile != nil {
			ss.csvTestFile.Close()
			ss.csvTestFile = nil
		}
		ss.csvTestWriter = nil
	}
}

// boolToStr helper
func boolToStr(b bool) string {
	if b {
		return "1"
	}
	return "0"
}
