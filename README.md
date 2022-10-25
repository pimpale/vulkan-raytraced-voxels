# Vulkan Raytraced Voxel Renderer

Not done yet.
It turns out that you can't enable the raytracing vulkan layers
when linking address sanitizer.

It fails on createDevice.

## How to build

Before you can run the program, you need to ensure that you have installed the Vulkan libraries and headers.

```bash
$ cd assets/shaders
$ ./compile.sh
$ cd ..
$ make
```

Run from the project root directory.

```bash
$ ./obj/vulkan-triangle-v2
```

## How to modify image assets
Block textures can be found in the `assets/blocks/` directory.
These textures are in the [farbfeld](http://tools.suckless.org/farbfeld/) format.
If you want to edit these files, you can use ImageMagick tools to convert them to png format,
and then back again to farbfeld format once you are done editing.

Example:
```bash
$ convert up.ff up.png
```

## Credits and Acknowledgments
Part of this project is based upon <https://vulkan-tutorial.com>.

I also referenced my prior example project: <https://github.com/pimpale/vulkan-triangle-v1>

### External Libraries
* `linmath.h`, which can be found at <https://github.com/datenwolf/linmath.h>
* `hashmap.c`, which can be found at <https://github.com/tidwall/hashmap.c>
* `open-simplex-noise.h`, which can be found at <https://github.com/smcameron/open-simplex-noise-in-c>
* `threadpool.h`, which can be found at <https://github.com/mbrossard/threadpool>

