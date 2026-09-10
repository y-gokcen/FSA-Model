module github.com/y-gokcen/FSA-Model

go 1.24.2

// Go 1.24 made math/rand.Seed a no-op, which silently un-seeded the network
// weight init (randx.SysRand falls back to the global source). Restore it so
// runs are reproducible from the run number.
godebug randseednop=0

require (
	cogentcore.org/core v0.3.10
	cogentcore.org/lab v0.1.0
	github.com/emer/emergent/v2 v2.0.0-dev0.1.7.0.20250307035120-1fc133d0d3ed
	github.com/emer/etensor v0.0.0-20250128231607-f3fea92f0b80
	github.com/emer/leabra/v2 v2.0.0-dev0.5.5.0.20250128232242-79e931d6fe3b
)

require (
	github.com/Bios-Marcel/wastebasket/v2 v2.0.3 // indirect
	github.com/BurntSushi/toml v1.3.2 // indirect
	github.com/Masterminds/vcs v1.13.3 // indirect
	github.com/alecthomas/chroma/v2 v2.13.0 // indirect
	github.com/anthonynsimon/bild v0.13.0 // indirect
	github.com/aymerick/douceur v0.2.0 // indirect
	github.com/chewxy/math32 v1.10.1 // indirect
	github.com/cogentcore/webgpu v0.0.0-20250118183535-3dd1436165cf // indirect
	github.com/dlclark/regexp2 v1.11.0 // indirect
	github.com/ericchiang/css v1.3.0 // indirect
	github.com/fsnotify/fsnotify v1.7.0 // indirect
	github.com/go-gl/glfw/v3.3/glfw v0.0.0-20240506104042-037f3cc74f2a // indirect
	github.com/gobwas/glob v0.2.3 // indirect
	github.com/goki/freetype v1.0.5 // indirect
	github.com/gomarkdown/markdown v0.0.0-20240930133441-72d49d9543d8 // indirect
	github.com/gorilla/css v1.0.1 // indirect
	github.com/h2non/filetype v1.1.3 // indirect
	github.com/hack-pad/go-indexeddb v0.3.2 // indirect
	github.com/hack-pad/hackpadfs v0.2.1 // indirect
	github.com/hack-pad/safejs v0.1.1 // indirect
	github.com/jinzhu/copier v0.4.0 // indirect
	github.com/mitchellh/go-homedir v1.1.0 // indirect
	github.com/pelletier/go-toml/v2 v2.1.2-0.20240227203013-2b69615b5d55 // indirect
	golang.org/x/exp v0.0.0-20250128182459-e0ece0dbea4c // indirect
	golang.org/x/image v0.18.0 // indirect
	golang.org/x/mod v0.22.0 // indirect
	golang.org/x/net v0.36.0 // indirect
	golang.org/x/sync v0.11.0 // indirect
	golang.org/x/sys v0.30.0 // indirect
	golang.org/x/text v0.22.0 // indirect
	golang.org/x/tools v0.29.0 // indirect
	gonum.org/v1/gonum v0.15.1 // indirect
)
